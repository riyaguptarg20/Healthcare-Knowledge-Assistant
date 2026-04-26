import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # inner product (cosine via normalized vectors)
        self.texts = []

    def add(self, embeddings, texts):
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query_embedding, k: int):
        query_embedding = np.array([query_embedding]).astype("float32")

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for idx, i in enumerate(indices[0]):
            if i < len(self.texts):
                results.append((self.texts[i], float(scores[0][idx])))

        return results