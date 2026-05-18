# ----- scholarsync/vectordb.py -----
"""
Vector database module for ScholarSync.

Responsibilities:
- Initialize HuggingFace sentence-transformer embeddings
- Initialize persistent ChromaDB vector store
- Log initialization steps
- Provide safe error handling for embedding/model failures

Tech:
- langchain-huggingface
- langchain-chroma
- chromadb (persistent)

Environment-controlled via Settings:
- EMBEDDING_MODEL
- CHROMA_PERSIST_DIR
- CHROMA_COLLECTION
"""

from __future__ import annotations

from typing import Any

from .config import Settings
from .logger import get_logger
from .utils import EmbeddingInitError

logger = get_logger("scholarsync.vectordb")


# -------------------------------------------------
# Embeddings Initialization
# -------------------------------------------------
def get_embeddings(settings: Settings) -> Any:
    """
    Initialize HuggingFace sentence-transformer embeddings.

    Uses:
        langchain_huggingface.HuggingFaceEmbeddings

    Raises:
        EmbeddingInitError on failure.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Initializing embeddings model: %s", settings.embedding_model)

        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model
        )

        logger.info("Embeddings initialized successfully.")
        return embeddings

    except Exception as e:
        logger.exception("Failed to initialize embeddings model.")
        raise EmbeddingInitError(
            f"Embedding model initialization failed. "
            f"Check EMBEDDING_MODEL='{settings.embedding_model}' and internet/cache."
        ) from e


# -------------------------------------------------
# Chroma Vector Store Initialization
# -------------------------------------------------
def get_vectorstore(settings: Settings, embeddings: Any):
    """
    Initialize persistent Chroma vector store.

    Uses:
        langchain_chroma.Chroma

    Persistence:
        Stored under settings.chroma_persist_dir

    Returns:
        Chroma vectorstore instance
    """
    try:
        from langchain_chroma import Chroma

        logger.info(
            "Initializing Chroma vectorstore | persist_dir=%s | collection=%s",
            settings.chroma_persist_dir,
            settings.chroma_collection,
        )

        from pathlib import Path
        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        # Clear stale Chroma client cache (no-op if not present).
        try:
            from chromadb.api.client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass
        vectorstore = Chroma(
            persist_directory=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection,
            embedding_function=embeddings,
        )

        logger.info("Chroma vectorstore initialized successfully.")
        return vectorstore

    except Exception as e:
        logger.exception("Failed to initialize Chroma vectorstore.")
        raise RuntimeError(
            f"Chroma initialization failed for "
            f"collection='{settings.chroma_collection}' "
            f"in persist_dir='{settings.chroma_persist_dir}'."
        ) from e
