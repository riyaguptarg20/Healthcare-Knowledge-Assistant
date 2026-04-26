from ragas.metrics import faithfulness, answer_relevancy

class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, question, answer, context):
        """
        LLM-based evaluation using RAGAS
        """
        try:
            results = {
                "faithfulness": faithfulness.score(answer, context),
                "relevance": answer_relevancy.score(answer, question)
            }
        except Exception as e:
            results = {
                "faithfulness": None,
                "relevance": None,
                "error": str(e)
            }

        return results


# ---------- Additional deterministic metrics ----------

def recall_at_k(retrieved_docs, ground_truth_docs):
    if not ground_truth_docs:
        return 0
    return len(set(retrieved_docs) & set(ground_truth_docs)) / len(ground_truth_docs)


def precision_at_k(retrieved_docs, ground_truth_docs):
    if not retrieved_docs:
        return 0
    return len(set(retrieved_docs) & set(ground_truth_docs)) / len(retrieved_docs)