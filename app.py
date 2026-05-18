# ----- app.py -----
# ScholarSync Streamlit UI (Upload → Build → Ask)
# - "Use only current upload" mode (default ON):
#     * Clears old UI-uploaded PDFs
#     * Resets Chroma index
#     * Ingests ONLY the newly uploaded PDFs
# - Manual index refresh (no background Chroma calls)
# - Instant theme toggle
# - Ingestion lock
# - RAG lazy initialization (only on ask)

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st
from dotenv import load_dotenv

from ingest import ingest as run_ingest
from scholarsync.config import Settings
from scholarsync.logger import get_logger
from scholarsync.rag import ScholarSyncRAG
from scholarsync.ui import apply_theme, chips_row, evidence_cards
from scholarsync.utils import GroqCallError, IndexNotFoundError

logger = get_logger("scholarsync.app")


# =========================================================
# Session State Initialization
# =========================================================
def _init_session():
    if "theme" not in st.session_state:
        st.session_state["theme"] = "Light"

    if "theme_toggle" not in st.session_state:
        st.session_state["theme_toggle"] = False

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "rag" not in st.session_state:
        st.session_state["rag"] = None

    if "ingesting" not in st.session_state:
        st.session_state["ingesting"] = False

    if "index_status" not in st.session_state:
        st.session_state["index_status"] = {"ok": False, "count": 0, "reason": "Not checked yet."}


# =========================================================
# Theme Toggle Callback
# =========================================================
def _on_theme_toggle():
    st.session_state["theme"] = "Dark" if st.session_state["theme_toggle"] else "Light"


# =========================================================
# Index Status (Manual only)
# =========================================================
def _index_status(settings: Settings) -> Dict[str, Any]:
    persist = Path(settings.chroma_persist_dir)

    if not persist.exists():
        return {"ok": False, "count": 0, "reason": "Persist directory not found."}

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(persist))
        try:
            coll = client.get_collection(settings.chroma_collection)
        except Exception:
            return {"ok": False, "count": 0, "reason": "Collection not found."}

        count = int(coll.count())
        if count <= 0:
            return {"ok": False, "count": count, "reason": "Collection empty."}

        return {"ok": True, "count": count, "reason": "OK"}
    except Exception as e:
        return {"ok": False, "count": 0, "reason": f"Index check failed: {e}"}


def _refresh_index_status(settings: Settings):
    st.session_state["index_status"] = _index_status(settings)


# =========================================================
# File helpers
# =========================================================
def _clear_dir(dir_path: Path) -> None:
    """Delete all files/subfolders under dir_path (keeps dir_path)."""
    if not dir_path.exists():
        return
    for p in dir_path.glob("*"):
        try:
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            logger.exception("Failed clearing path: %s", str(p))


def _save_uploaded_pdfs(uploaded_files, base_dir: Path) -> int:
    base_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for uf in uploaded_files:
        name = Path(uf.name).name  # prevent path traversal
        if not name.lower().endswith(".pdf"):
            continue

        target = base_dir / name
        if target.exists():
            target = base_dir / f"{target.stem}_{uuid.uuid4().hex[:8]}{target.suffix}"

        try:
            target.write_bytes(uf.getbuffer())
            saved += 1
        except Exception:
            logger.exception("Failed saving PDF: %s", name)

    return saved


# =========================================================
# Sidebar UI
# =========================================================
def _build_sidebar(settings: Settings) -> Dict[str, Any]:
    st.sidebar.markdown("## 📚 ScholarSync")
    st.sidebar.caption("Upload PDFs → Build index → Ask questions (grounded + citations)")
    st.sidebar.divider()

    # Theme
    st.sidebar.markdown("### 🌓 Theme")
    st.sidebar.toggle("Dark mode", key="theme_toggle", on_change=_on_theme_toggle)
    st.sidebar.divider()

    # Index
    st.sidebar.markdown("### 🗂️ Index")

    upload_dir = Path("data/pdfs/ui_uploads")

    uploaded = st.sidebar.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="These PDFs will be indexed for Q&A.",
    )

    #  Default ON: only current upload should be used
    isolate_mode = st.sidebar.toggle(
        "Use only current upload (recommended)",
        value=True,
        help="When ON: clears old uploads, resets index, and answers only from the latest uploaded PDFs.",
    )

    reset_index = st.sidebar.toggle(
        "Reset index (force rebuild)",
        value=False,
        help="Deletes existing Chroma index before rebuilding.",
        disabled=isolate_mode,  # isolate already forces reset
    )

    c1, c2 = st.sidebar.columns(2)
    refresh_clicked = c1.button("Refresh", use_container_width=True)

    build_clicked = c2.button(
    "Build",
    use_container_width=True,
    disabled=st.session_state["ingesting"]
    )

    if refresh_clicked:
        _refresh_index_status(settings)

    status = st.session_state["index_status"]

    if status["ok"]:
        st.sidebar.success(f"✅ Ready • {status['count']} chunks")
    else:
        st.sidebar.warning(f"⚠ {status['reason']}")

    # Ingestion action (guarded)
    if build_clicked:
        if not uploaded:
            st.sidebar.error("Upload at least one PDF first.")
        else:
            st.session_state["ingesting"] = True
            try:
                #  isolate_mode guarantees: no old PDFs + no old chunks
                effective_reset = True if isolate_mode else bool(reset_index)

                if isolate_mode:
                    # remove old UI uploads so ONLY new PDFs exist
                    _clear_dir(upload_dir)

                saved = _save_uploaded_pdfs(uploaded, upload_dir)
                if saved <= 0:
                    st.sidebar.error("No PDFs saved.")
                else:
                    with st.sidebar.status("Indexing PDFs…", expanded=True):
                        # ✅ ingest ONLY UI upload folder (not the whole data/pdfs)
                        run_ingest(upload_dir, reset=effective_reset)

                    # refresh status once (manual)
                    _refresh_index_status(settings)

                    # reset RAG so it reloads the fresh index
                    st.session_state["rag"] = None

                    if isolate_mode:
                        st.sidebar.success("✅ Built index (current upload only).")
                    else:
                        st.sidebar.success("✅ Built/updated index (appended).")

            except Exception:
                logger.exception("UI ingestion failed.")
                st.sidebar.error("Ingestion failed. Check logs/scholarsync.log")
            finally:
                st.session_state["ingesting"] = False

    st.sidebar.divider()

    # Retrieval knobs
    st.sidebar.markdown("### 🔎 Retrieval")
    top_k = st.sidebar.number_input("TOP_K", 1, 50, int(settings.top_k))
    fetch_k = st.sidebar.number_input("FETCH_K", 1, 200, int(settings.fetch_k))
    max_ctx = st.sidebar.number_input("MAX_CONTEXT_TOKENS", 256, 20000, int(settings.max_context_tokens))

    st.sidebar.divider()
    st.sidebar.markdown("### ⚡ Model")
    st.sidebar.code(settings.groq_model, language="text")

    st.sidebar.divider()
    if st.sidebar.button(" Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.toast("Chat cleared.", icon="🧹")

    return {
        "top_k": int(top_k),
        "fetch_k": int(fetch_k),
        "max_context_tokens": int(max_ctx),
        "status": status,
        "isolate_mode": isolate_mode,
    }


# =========================================================
# Lazy RAG Init
# =========================================================
def _ensure_rag(settings: Settings) -> Optional[ScholarSyncRAG]:
    if st.session_state["rag"] is not None:
        return st.session_state["rag"]

    if not settings.groq_api_key:
        st.error("🔑 Missing GROQ_API_KEY in .env")
        st.info("Copy .env.example → .env and paste your Groq key.")
        return None

    try:
        with st.spinner("Initializing RAG…"):
            rag = ScholarSyncRAG(settings)
        st.session_state["rag"] = rag
        return rag
    except Exception:
        logger.exception("RAG init failed.")
        st.error("Failed initializing RAG. Check logs/scholarsync.log")
        return None


# =========================================================
# Main
# =========================================================
def main():
    load_dotenv()
    st.set_page_config(page_title="ScholarSync", page_icon="📚", layout="wide")

    _init_session()
    apply_theme(st.session_state["theme"])

    try:
        settings = Settings.from_env()
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.markdown("# 📚 ScholarSync")
    st.markdown("<div class='muted'>Upload PDFs → Build index → Ask questions.</div>", unsafe_allow_html=True)

    sidebar = _build_sidebar(settings)

    # Guidance if index missing
    if not sidebar["status"]["ok"]:
        st.warning("⚠ Index not ready. Upload PDFs and click Build.")
        with st.expander("How it works", expanded=False):
            st.markdown(
                """
- **Use only current upload** (recommended): answers come only from latest PDFs.
- Click **Build** after uploading.
- Then ask your question in the chat box below.
                """.strip()
            )

    # Chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    q = st.chat_input("Ask a question about your uploaded PDFs...")

    if q:
        st.session_state["messages"].append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        rag = _ensure_rag(settings)
        if rag is None:
            return

        # Apply runtime knobs
        rag.update_knobs(
            sidebar["top_k"],
            sidebar["fetch_k"],
            sidebar["max_context_tokens"],
        )

        try:
            result = rag.ask(q)

            answer = result["answer"]
            st.session_state["messages"].append({"role": "assistant", "content": answer})

            with st.chat_message("assistant"):
                st.markdown(answer)

                dbg = result["debug"]
                timings = dbg["timings"]

                chips_row(
                    [
                        {"label": "Total", "value": f"{timings['total_s']:.3f}s"},
                        {"label": "Retrieve", "value": f"{timings['retrieve_s']:.3f}s"},
                        {"label": "Pack", "value": f"{timings['pack_s']:.3f}s"},
                        {"label": "Generate", "value": f"{timings['gen_s']:.3f}s"},
                    ]
                )

                st.markdown("### 📌 Evidence")
                evidence_cards(result["sources"])

        except IndexNotFoundError:
            st.error("Index not found. Upload PDFs and click Build.")
        except GroqCallError:
            st.error("Groq request failed.")
        except Exception:
            logger.exception("Unhandled UI query error.")
            st.error("Unexpected error. Check logs/scholarsync.log")


if __name__ == "__main__":
    main()
