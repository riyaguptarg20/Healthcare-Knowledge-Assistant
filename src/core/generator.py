import requests
import logging


class Generator:
    def __init__(self, model_name="llama3"):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/generate"

    def generate(self, query, context):
        if not context:
            return "I don't know based on the provided information."

        prompt = f"""
Answer ONLY using the provided context.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}
"""

        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )

            response.raise_for_status()
            return response.json().get("response", "").strip()

        except Exception as e:
            logging.error(f"Ollama generation failed: {e}")
            return "Error generating response."