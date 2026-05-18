# ----- ingest.py -----
# ScholarSync PDF Ingestion CLI
#
# What it does:
# 1) Recursively finds PDFs under data/pdfs/ (or --pdf_dir)
# 2) Loads pages using PyPDFLoader (skips corrupt PDFs safely)
# 3) Chunks text (semantic chunking preferred, else recursive splitter)
# 4) Embeds + stores chunks into a persistent ChromaDB collection
# 5) Logs ingestion stats + timings + batch insert timings
#
# Run:
#   python ingest.py
#   python ingest.py --reset
#   python ingest.py --pdf_dir path/to/pdfs --reset

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from tqdm import tqdm

from scholarsync.chunking import semantic_or_recursive_split
from scholarsync.config import Settings
from scholarsync.logger import get_logger
from scholarsync.utils import NoPDFsFoundError
from scholarsync.vectordb import get_embeddings, get_vectorstore

logger = get_logger("scholarsync.ingest")


# -----------------------------
# Helper: Find PDFs recursively
# -----------------------------
def _find_pdfs(pdf_dir: Path) -> List[Path]:
    """Return list of all .pdf files under pdf_dir recursively."""
    return [p for p in pdf_dir.rglob("*.pdf") if p.is_file()]


# -----------------------------
# Helper: Load PDF safely
# -----------------------------
def _load_pdf_safely(pdf_path: Path) -> List[Document]:
    """
    Load a PDF using PyPDFLoader (per your requirement).
    If the PDF is corrupt/unreadable, return [] and log warning, but do not crash ingestion.

    Note: PyPDFLoader returns one Document per page.
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()  # per-page documents

        # Normalize metadata: store source as filename only (not full path)
        for d in docs:
            d.metadata = dict(d.metadata or {})
            d.metadata["source"] = pdf_path.name  # required: filename
            # page is 0-index in PyPDFLoader; keep as is
            if "page" not in d.metadata:
                d.metadata["page"] = 0

        return docs

    except Exception as e:
        logger.warning("Skipping corrupt/unreadable PDF: %s (%s)", str(pdf_path), str(e))
        return []


# -----------------------------
# Main ingestion function
# -----------------------------
def ingest(pdf_dir: Path, reset: bool) -> int:
    """
    End-to-end ingestion:
    - load settings
    - optionally reset chroma persist dir
    - load pdf pages (skip corrupt)
    - chunk
    - add metadata chunk_id
    - batch insert into persistent chroma
    """
    settings = Settings.from_env()

    pdf_dir = pdf_dir.resolve()
    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True, exist_ok=True)

    # Reset requested: delete existing persist dir
    persist_dir = Path(settings.chroma_persist_dir)
    if reset and persist_dir.exists():
        logger.info("Reset requested: removing existing Chroma directory: %s", str(persist_dir))
        shutil.rmtree(persist_dir, ignore_errors=True)
        # Critical: clear ChromaDB's in-process client cache.
        # Without this, the next Chroma(...) call returns a stale client whose
        # SQLite handle points at the directory we just deleted, causing
        # "sqlite3.OperationalError: unable to open database file".
        try:
            from chromadb.api.client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
        except Exception:
            logger.debug("Could not clear Chroma SharedSystemClient cache (non-fatal).")

    # Chroma needs the persist dir to exist before it can create the SQLite file.
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Find PDFs
    pdfs = _find_pdfs(pdf_dir)
    if not pdfs:
        raise NoPDFsFoundError(f"No PDFs found in {pdf_dir}. Add PDFs to data/pdfs/ and try again.")

    logger.info("Ingestion start | pdf_dir=%s | pdf_count=%d", str(pdf_dir), len(pdfs))

    # Initialize embeddings (also used by SemanticChunker if available)
    try:
        embeddings = get_embeddings(settings)
    except Exception:
        logger.exception("Embedding initialization failed.")
        raise

    # Load all pages across PDFs (skip corrupt PDFs)
    all_pages: List[Document] = []
    page_count = 0
    skipped = 0

    for p in tqdm(pdfs, desc="Loading PDFs", unit="pdf"):
        docs = _load_pdf_safely(p)
        if not docs:
            skipped += 1
            continue
        all_pages.extend(docs)
        page_count += len(docs)

    # If nothing loaded successfully, fail with friendly message
    if not all_pages:
        raise NoPDFsFoundError(
            "All PDFs failed to load (possibly corrupt). Please replace PDFs in data/pdfs/."
        )

    logger.info(
        "PDF load complete | loaded_pdfs=%d skipped_pdfs=%d pages=%d",
        len(pdfs) - skipped,
        skipped,
        page_count,
    )

    # Chunking step
    t0 = time.perf_counter()
    chunks = semantic_or_recursive_split(all_pages, embeddings=embeddings, base_chunk_size=900)
    chunk_s = time.perf_counter() - t0

    # Add required metadata for each chunk: chunk_id, source filename, page
    for i, c in enumerate(chunks):
        meta = dict(c.metadata or {})
        src = meta.get("source", "unknown.pdf")
        page = meta.get("page", 0)  # 0-index
        meta["chunk_id"] = f"{src}|p{page}|c{i}"
        c.metadata = meta

    logger.info("Chunking complete | chunks=%d | time=%.3fs", len(chunks), chunk_s)

    # Initialize / open persistent Chroma vectorstore
    vs = get_vectorstore(settings, embeddings)

    # Batch insert for better latency + logging
    batch_size = 64
    insert_start = time.perf_counter()
    inserted = 0

    for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing", unit="batch"):
        batch = chunks[i : i + batch_size]
        bt = time.perf_counter()
        vs.add_documents(batch)
        inserted += len(batch)
        logger.info("Batch insert | size=%d | time=%.3fs", len(batch), time.perf_counter() - bt)

    insert_s = time.perf_counter() - insert_start

    # Persist (if available; chroma persists automatically, but some versions expose persist())
    try:
        persist_fn = getattr(vs, "persist", None)
        if callable(persist_fn):
            vs.persist()
    except Exception:
        logger.warning("Vectorstore persist() not available; relying on Chroma persistence.")

    logger.info(
        "Ingestion end | pdfs=%d pages=%d chunks=%d inserted=%d | chunk_time=%.3fs insert_time=%.3fs",
        len(pdfs),
        page_count,
        len(chunks),
        inserted,
        chunk_s,
        insert_s,
    )
    return 0


# -----------------------------
# CLI entrypoint
# -----------------------------
def main() -> int:
    """
    CLI wrapper that:
    - loads .env
    - parses args
    - runs ingestion
    - returns exit codes:
        0 = success
        1 = unhandled error
        2 = user/config error (no PDFs, invalid env)
    """
    load_dotenv()

    parser = argparse.ArgumentParser(description="ScholarSync ingestion: load PDFs and build Chroma index.")
    parser.add_argument("--pdf_dir", type=str, default="data/pdfs", help="Directory containing PDFs (recursive).")
    parser.add_argument("--reset", action="store_true", help="Delete existing Chroma persist directory before ingest.")
    args = parser.parse_args()

    try:
        return ingest(Path(args.pdf_dir), reset=args.reset)

    except NoPDFsFoundError as e:
        logger.error(str(e))
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    except ValueError as e:
        logger.error("Config validation error: %s", str(e))
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    except Exception as e:
        logger.exception("Unhandled ingestion error.")
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
