import time
import hashlib
import logging

from core.context_builder import build_context
from caching.redis_cache import get_cache, set_cache


class RAGPipeline:
    def __init__(self, retriever, reranker, generator, config):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.config = config

    def _cache_key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def run(self, query: str):
        start_time = time.time()

        logging.info(f"Query received: {query}")

        cache_key = self._cache_key(query)

        # -------- Cache Check --------
        cached = get_cache(cache_key)
        if cached:
            logging.info("Cache hit")

            try:
                cached_value = cached.decode() if isinstance(cached, bytes) else cached
            except Exception:
                cached_value = str(cached)

            return {
                "answer": cached_value,
                "latency": 0,
                "cached": True
            }

        logging.info("Cache miss")

        # -------- Retrieval --------
        docs = self.retriever.retrieve(query, self.config.top_k)
        logging.info(f"Retrieved {len(docs)} documents")

        if not docs:
            return {
                "answer": "No relevant information found.",
                "latency": 0,
                "cached": False
            }

        # -------- Reranking --------
        docs = self.reranker.rerank(query, docs, self.config.final_k)

        # -------- Context Building --------
        context = build_context(docs, self.config.max_tokens)

        # -------- Generation --------
        try:
            answer = self.generator.generate(query, context)
        except Exception as e:
            logging.error(f"LLM generation failed: {e}")
            return {
                "answer": "Error generating response. Please try again.",
                "latency": 0,
                "cached": False
            }

        latency = time.time() - start_time
        logging.info(f"Latency: {latency:.2f}s")

        # -------- Cache Result --------
        set_cache(cache_key, answer)

        return {
            "answer": answer,
            "latency": latency,
            "cached": False,
            "documents_used": docs
        }