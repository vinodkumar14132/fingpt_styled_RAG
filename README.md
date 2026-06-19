# FinGPT Styled RAG Assistant

A Retrieval-Augmented Generation (RAG) application built using:

- LangChain
- ChromaDB
- HuggingFace Embeddings
- FastAPI
- LangGraph
- Multi-Agent Architecture

## Features

- PDF Ingestion
- Vector Database Storage
- Semantic Search
- RAG Pipeline
- Routing Agents
- Specialist Agents
- Human Approval Workflow
- Memory-enabled Agents

## Installation

```bash
git clone <repo-url>
cd fingpt-styled-rag

pip install -r requirements.txt

python src/rag_pipeline.py
uvicorn app:app --reload
http://127.0.0.1:8000/ask?question=What is TCS strategy?