# ----- scholarsync/chunking.py -----
"""
Document chunking module for ScholarSync (UPDATED for speed + stability).

✅ Changes in this version (based on your logs):
1) SemanticChunker is VERY slow on CPU (you saw 150–265 seconds).
   So we add an ENV toggle to control it:
      USE_SEMANTIC_CHUNKING=1  (enable)
      USE_SEMANTIC_CHUNKING=0  (disable, default = 0 for low latency)

2) Fallback to RecursiveCharacterTextSplitter is always available and fast.

3) Dynamic overlap (~15%) bounded (80–220) is preserved.

4) If SemanticChunker is used, we do light overlap stitching between adjacent chunks.

Env var (optional):
    USE_SEMANTIC_CHUNKING=0   # recommended on Windows CPU for speed
"""

from __future__ import annotations

import os
from typing import List

from langchain_core.documents import Document

from .logger import get_logger

logger = get_logger("scholarsync.chunking")


# -------------------------------------------------
# Dynamic Overlap Calculation
# -------------------------------------------------
def _dynamic_overlap(chunk_size: int) -> int:
    """
    Compute dynamic overlap (~15% of chunk size),
    bounded between 80 and 220 characters.
    """
    overlap = int(chunk_size * 0.15)
    return max(80, min(220, overlap))


# -------------------------------------------------
# Overlap Stitching for Semantic Chunker
# -------------------------------------------------
def _stitch_overlap(chunks: List[Document], overlap_chars: int) -> List[Document]:
    """
    Adds a small prefix overlap from the previous chunk
    to improve contextual continuity.

    Only applied to semantic chunks.
    """
    if not chunks or overlap_chars <= 0:
        return chunks

    stitched: List[Document] = []
    prev_text = ""

    for i, doc in enumerate(chunks):
        text = doc.page_content or ""

        if i == 0:
            stitched.append(doc)
        else:
            prefix = prev_text[-overlap_chars:] if prev_text else ""
            merged_text = (prefix + "\n" + text).strip()

            stitched.append(
                Document(
                    page_content=merged_text,
                    metadata=dict(doc.metadata),
                )
            )

        prev_text = text

    return stitched


# -------------------------------------------------
# Main Chunking Function
# -------------------------------------------------
def semantic_or_recursive_split(
    docs: List[Document],
    embeddings,
    base_chunk_size: int = 900,
) -> List[Document]:
    """
    Perform document splitting using:

    1) SemanticChunker (optional; controlled by env USE_SEMANTIC_CHUNKING)
    2) RecursiveCharacterTextSplitter (default)

    Parameters:
        docs: list of LangChain Documents
        embeddings: embedding model (required for semantic chunking)
        base_chunk_size: approximate chunk size in characters

    Returns:
        List of chunked Documents
    """
    if not docs:
        logger.warning("No documents provided for chunking.")
        return []

    overlap = _dynamic_overlap(base_chunk_size)

    # -------------------------------------------------
    # ENV Toggle for Semantic Chunking
    # -------------------------------------------------
    use_semantic = os.getenv("USE_SEMANTIC_CHUNKING", "0").strip().lower() in ("1", "true", "yes", "y")

    if use_semantic:
        # -------------------------------------------------
        # Try Semantic Chunker (preferred when enabled)
        # -------------------------------------------------
        try:
            from langchain_experimental.text_splitter import SemanticChunker

            logger.info(
                "Chunking strategy: SemanticChunker (enabled) | base_chunk_size=%d | overlap≈%d",
                base_chunk_size,
                overlap,
            )

            splitter = SemanticChunker(embeddings=embeddings)
            chunks = splitter.split_documents(docs)

            # Apply small stitching overlap for continuity
            chunks = _stitch_overlap(chunks, overlap_chars=overlap)

            logger.info(
                "Semantic chunking completed | input_docs=%d | output_chunks=%d",
                len(docs),
                len(chunks),
            )
            return chunks

        except Exception as e:
            logger.warning(
                "SemanticChunker enabled but failed (%s). Falling back to recursive splitter.",
                str(e),
            )

    # -------------------------------------------------
    # Default: Recursive Character Splitter (FAST)
    # -------------------------------------------------
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        logger.info(
            "Chunking strategy: RecursiveCharacterTextSplitter (default) | chunk_size=%d | overlap=%d",
            base_chunk_size,
            overlap,
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=base_chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(docs)

        logger.info(
            "Recursive chunking completed | input_docs=%d | output_chunks=%d",
            len(docs),
            len(chunks),
        )

        return chunks

    except Exception:
        logger.exception("Recursive chunking failed.")
        raise
