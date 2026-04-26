"""
src/retrieval/hybrid_search.py

Hybrid retrieval combining:
  1. Dense vector search  (semantic embeddings)
  2. Sparse BM25 search   (keyword matching)
  3. Reciprocal Rank Fusion (RRF) for score combination

Why hybrid beats either alone:
  - Dense: handles paraphrasing, synonyms, semantic similarity
  - Sparse: handles exact drug names, dosages, rare terms
  - Fusion: consistently outperforms either by 10-15% on MRR@10

This is the single biggest differentiator in a RAG system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from src.ingestion.chunking import Chunk


# ─── Result Model ─────────────────────────────────────────────────────────────

@dataclass(order=True)
class RetrievalResult:
    """A single retrieved chunk with fusion scoring."""
    sort_index: float = field(init=False, repr=False)

    chunk:      Chunk
    score:      float           # final fusion score (higher = more relevant)
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float = 0.0
    sparse_score: float = 0.0

    def __post_init__(self) -> None:
        self.sort_index = -self.score   # enable sorted() → descending by score

    @property
    def source(self) -> str:
        return self.chunk.metadata.get("source", "unknown")

    @property
    def page(self) -> Any:
        return self.chunk.metadata.get("page", "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "content":      self.chunk.content,
            "score":        round(self.score, 4),
            "dense_rank":   self.dense_rank,
            "sparse_rank":  self.sparse_rank,
            "source":       self.source,
            "page":         self.page,
            "chunk_index":  self.chunk.chunk_index,
            "doc_id":       self.chunk.doc_id,
            "metadata":     self.chunk.metadata,
        }


# ─── BM25 Index ───────────────────────────────────────────────────────────────

class BM25Index:
    """
    Thin wrapper around rank-bm25 with pharma-aware tokenization.
    Handles: drug names with digits (e.g. "COVID-19"), dosages ("5mg"),
             and mixed-case medical acronyms.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._tokenized = [self._tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized)
        logger.debug(f"[BM25Index] Built index over {len(chunks)} chunks")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Pharma-aware tokenizer:
        - Lowercase
        - Keep alphanumeric + hyphens (drug names)
        - Split on whitespace and common delimiters
        """
        import re
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)
        return tokens

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Return (chunk_index, bm25_score) pairs, descending by score."""
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        # Get top_k indices sorted descending
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


# ─── Reciprocal Rank Fusion ───────────────────────────────────────────────────

def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
    weights: list[float] | None = None,
) -> dict[str, float]:
    """
    Combine multiple ranked lists via RRF.

    RRF(d) = Σ  weight_i / (k + rank_i(d))

    k=60 is the standard constant from Cormack et al. (2009).
    Higher k → more weight to lower-ranked items (smoother blending).

    Args:
        rankings: list of [(doc_id, score), ...] sorted descending
        k:        RRF constant
        weights:  per-ranking weights (default uniform)

    Returns:
        {doc_id: rrf_score} mapping
    """
    if weights is None:
        weights = [1.0] * len(rankings)

    rrf_scores: dict[str, float] = {}
    for rank_list, w in zip(rankings, weights):
        for rank, (doc_id, _) in enumerate(rank_list, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + w / (k + rank)

    return rrf_scores


# ─── Hybrid Retriever ─────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Orchestrates dense + sparse retrieval with RRF fusion.

    Parameters
    ----------
    vector_store :
        Object with a `.similarity_search_with_score(query, k)` method
        returning list[(Chunk, float)].
    chunks :
        All indexed chunks (needed to build BM25 index).
    alpha :
        Weight for dense component in RRF (1-alpha for sparse).
        alpha=1.0 → pure dense; alpha=0.0 → pure BM25.
    top_k :
        Final number of results to return.
    similarity_threshold :
        Minimum dense similarity to include a result.
    """

    def __init__(
        self,
        vector_store: Any,
        chunks: list[Chunk],
        alpha: float = 0.6,
        top_k: int = 10,
        similarity_threshold: float = 0.35,
    ) -> None:
        self.vector_store = vector_store
        self.alpha = alpha
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        logger.info("[HybridRetriever] Building BM25 index...")
        self._bm25_index = BM25Index(chunks)
        self._chunk_map: dict[str, Chunk] = {c.doc_id: c for c in chunks}
        logger.success(f"[HybridRetriever] Ready. α={alpha}, top_k={top_k}")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Execute hybrid retrieval for a single query.

        Returns results sorted by fusion score (descending).
        """
        k = top_k or self.top_k
        fetch_k = min(k * 3, 50)  # over-fetch before fusion

        # ── 1. Dense retrieval ────────────────────────────────
        dense_raw = self.vector_store.similarity_search_with_score(query, k=fetch_k)
        dense_ranking: list[tuple[str, float]] = []
        dense_score_map: dict[str, float] = {}
        for chunk, score in dense_raw:
            if score >= self.similarity_threshold:
                dense_ranking.append((chunk.doc_id, score))
                dense_score_map[chunk.doc_id] = score

        # ── 2. Sparse BM25 retrieval ──────────────────────────
        sparse_raw = self._bm25_index.search(query, top_k=fetch_k)
        sparse_ranking: list[tuple[str, float]] = []
        sparse_score_map: dict[str, float] = {}
        for idx, score in sparse_raw:
            chunk = self._bm25_index.chunks[idx]
            sparse_ranking.append((chunk.doc_id, score))
            sparse_score_map[chunk.doc_id] = score

        # ── 3. RRF Fusion ─────────────────────────────────────
        fused = reciprocal_rank_fusion(
            [dense_ranking, sparse_ranking],
            weights=[self.alpha, 1.0 - self.alpha],
        )

        # ── 4. Rank → Results ─────────────────────────────────
        ranked_ids = sorted(fused, key=lambda x: fused[x], reverse=True)[:k]

        results: list[RetrievalResult] = []
        for rank, doc_id in enumerate(ranked_ids, start=1):
            chunk = self._chunk_map.get(doc_id)
            if chunk is None:
                continue
            dense_rank  = next((i+1 for i, (d, _) in enumerate(dense_ranking) if d == doc_id), None)
            sparse_rank = next((i+1 for i, (d, _) in enumerate(sparse_ranking) if d == doc_id), None)
            results.append(RetrievalResult(
                chunk=chunk,
                score=fused[doc_id],
                dense_rank=dense_rank,
                sparse_rank=sparse_rank,
                dense_score=dense_score_map.get(doc_id, 0.0),
                sparse_score=sparse_score_map.get(doc_id, 0.0),
            ))

        logger.debug(
            f"[HybridRetriever] Query='{query[:50]}' → "
            f"{len(dense_ranking)} dense, {len(sparse_ranking)} sparse, "
            f"{len(results)} fused"
        )
        return results

    def retrieve_with_expansion(
        self,
        query: str,
        expanded_queries: list[str],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve over original + expanded queries, fuse all results.
        Query expansion increases recall for ambiguous queries.
        """
        all_rankings: list[list[tuple[str, float]]] = []
        weights: list[float] = []

        # Original query gets 2x weight
        original_results = self.retrieve(query, top_k=top_k)
        original_ranking = [(r.chunk.doc_id, r.score) for r in original_results]
        all_rankings.append(original_ranking)
        weights.append(2.0)

        for eq in expanded_queries:
            eq_results = self.retrieve(eq, top_k=top_k)
            eq_ranking = [(r.chunk.doc_id, r.score) for r in eq_results]
            all_rankings.append(eq_ranking)
            weights.append(1.0)

        fused = reciprocal_rank_fusion(all_rankings, weights=weights)
        k = top_k or self.top_k
        ranked_ids = sorted(fused, key=lambda x: fused[x], reverse=True)[:k]

        results: list[RetrievalResult] = []
        for doc_id in ranked_ids:
            chunk = self._chunk_map.get(doc_id)
            if chunk:
                results.append(RetrievalResult(chunk=chunk, score=fused[doc_id]))

        return results