import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fastapi import FastAPI
from core.config import Config
from core.reranker import Reranker
from core.generator import Generator
from core.pipeline import RAGPipeline
from scripts.ingest import ingest  # make sure this exists

logging.basicConfig(level=logging.INFO)

app = FastAPI()
pipeline = None


def load_your_data():
    """
    Replace this with real ingestion later
    """
    return [
        "Hypertension is high blood pressure.",
        "Aspirin can cause gastrointestinal bleeding.",
        "Diabetes is a chronic metabolic disorder.",
        "Insulin regulates blood sugar levels."
    ]


@app.on_event("startup")
def startup():
    global pipeline

    logging.info("Starting RAG system...")

    texts = load_your_data()

    vector_store, retriever = ingest(texts)

    reranker = Reranker()
    generator = Generator()
    config = Config()

    pipeline = RAGPipeline(retriever, reranker, generator, config)

    logging.info("RAG pipeline initialized successfully")


@app.get("/query")
def query(q: str):
    if pipeline is None:
        return {"error": "Pipeline not initialized"}

    return pipeline.run(q)