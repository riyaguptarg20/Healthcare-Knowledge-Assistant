from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, docs, top_k=5):
        if not docs:
            return []

        pairs = [[query, doc] for doc in docs]
        scores = self.model.predict(pairs, batch_size=16)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked[:top_k]]