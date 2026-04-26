from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class HybridRetriever:
    def __init__(self, vector_store, embedder, alpha=0.75):
        self.vector_store = vector_store
        self.embedder = embedder
        self.alpha = alpha

        self.tfidf = TfidfVectorizer()
        self.corpus = []
        self.tfidf_matrix = None

    def fit_sparse(self, texts):
        """
        Fit TF-IDF on corpus
        """
        self.corpus = texts
        self.tfidf_matrix = self.tfidf.fit_transform(texts)

    def retrieve(self, query, top_k=20):
        """
        Hybrid retrieval: Dense + Sparse with normalization
        """

        # ✅ Safety check
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF not initialized. Call fit_sparse() first.")

        # -------- Dense retrieval --------
        dense_vec = self.embedder.encode(query)[0]
        dense_results = self.vector_store.search(dense_vec, top_k)

        dense_dict = {doc: score for doc, score in dense_results}

        # -------- Sparse retrieval --------
        sparse_vec = self.tfidf.transform([query])
        sparse_scores = (self.tfidf_matrix @ sparse_vec.T).toarray().flatten()

        # -------- Normalize scores --------
        if len(dense_dict) > 0:
            dense_values = np.array(list(dense_dict.values()))
            dense_min, dense_max = dense_values.min(), dense_values.max()
        else:
            dense_min, dense_max = 0, 1

        sparse_min, sparse_max = sparse_scores.min(), sparse_scores.max()

        combined_scores = {}

        for idx, doc in enumerate(self.corpus):
            sparse_score = sparse_scores[idx]

            # Normalize sparse
            if sparse_max - sparse_min > 1e-6:
                sparse_score = (sparse_score - sparse_min) / (sparse_max - sparse_min)

            # Normalize dense
            dense_score = dense_dict.get(doc, 0)
            if dense_max - dense_min > 1e-6:
                dense_score = (dense_score - dense_min) / (dense_max - dense_min)

            # Combine
            score = self.alpha * dense_score + (1 - self.alpha) * sparse_score
            combined_scores[doc] = score

        # -------- Rank --------
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked[:top_k]]