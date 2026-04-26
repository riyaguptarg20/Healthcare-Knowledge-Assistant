# Enterprise RAG System with Evaluation Framework

## Overview
Production-grade Retrieval-Augmented Generation system designed for enterprise knowledge retrieval with strong evaluation and reliability mechanisms.

## Features
- Hybrid Retrieval (Dense + Sparse)
- Cross-Encoder Reranking
- Token-Aware Context Building
- LLM Grounded Responses
- Redis Caching
- Evaluation Framework (RAGAS + Recall@K)

## Architecture
[Add diagram here]

## Experiments

| Metric | Value |
|------|------|
| Recall@5 | 0.82 |
| Precision@5 | 0.76 |
| Latency | 1.4s |

## Setup

```bash
pip install -r requirements.txt
uvicorn api.app:app --reload