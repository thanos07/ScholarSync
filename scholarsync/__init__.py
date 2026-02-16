# ----- scholarsync/__init__.py -----
"""
ScholarSync
===========

AI-powered Retrieval-Augmented Generation (RAG) system for
context-aware Q&A over a folder of PDFs.

Core features:
- LangChain + LCEL modular RAG pipeline
- Groq LLM (low-latency)
- Chroma persistent vector store
- HuggingFace sentence-transformer embeddings
- Semantic chunking (fallback to recursive)
- Strict grounding + citation enforcement
- Streamlit UI with Light/Dark runtime theme
- Production logging (RotatingFileHandler)

Package Structure:
    scholarsync/
        config.py      -> Environment config + validation
        logger.py      -> Centralized logging setup
        chunking.py    -> Document chunking logic
        vectordb.py    -> Embeddings + Chroma setup
        prompts.py     -> Strict RAG prompt template
        utils.py       -> Packing, similarity, helpers
        rag.py         -> LCEL RAG orchestration
        ui.py          -> UI styling + components

This file:
- Marks the directory as a Python package
- Exposes core modules
- Defines package version
"""

__all__ = [
    "config",
    "logger",
    "chunking",
    "vectordb",
    "prompts",
    "utils",
    "rag",
    "ui",
]

__version__ = "1.0.0"
