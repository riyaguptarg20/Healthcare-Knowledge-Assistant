from dataclasses import dataclass

@dataclass
class Config:
    # Retrieval
    top_k: int = 20
    final_k: int = 5
    alpha: float = 0.75  # hybrid weight (dense vs sparse)

    # LLM
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0

    # Context
    max_tokens: int = 3000

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Debug / Logs
    enable_logging: bool = True