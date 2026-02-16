# ----- scholarsync/rag.py -----
"""
ScholarSync RAG Pipeline (LCEL-based)

Fix included:
- LCEL RunnablePassthrough.assign passes full input dict.
- Retrieval now correctly extracts x["question"] (string) before embedding.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from .config import Settings
from .logger import get_logger
from .prompts import strict_rag_prompt
from .utils import (
    GroqCallError,
    IndexNotFoundError,
    MissingAPIKeyError,
    PackedContext,
    ensure_citations,
    evidence_list,
    pack_context,
)
from .vectordb import get_embeddings, get_vectorstore

logger = get_logger("scholarsync.rag")


class ScholarSyncRAG:
    def __init__(self, settings: Settings):
        self.settings = settings

        # embeddings + vectorstore
        self.embeddings = get_embeddings(settings)
        self.vectorstore = get_vectorstore(settings, self.embeddings)

        # strict prompt
        self.prompt = strict_rag_prompt()

        # require key for querying
        if not self.settings.groq_api_key:
            raise MissingAPIKeyError(
                "Missing GROQ_API_KEY. Set it in your environment or .env file before querying."
            )

        # Groq LLM
        try:
            from langchain_groq import ChatGroq

            self.llm = ChatGroq(
                api_key=self.settings.groq_api_key,
                model=self.settings.groq_model,
                temperature=0.2,
            )
        except Exception:
            logger.exception("Failed to initialize Groq client.")
            raise

        self._build_chain()

    # -----------------------------
    # Runtime knobs (UI)
    # -----------------------------
    def update_knobs(self, top_k: int, fetch_k: int, max_context_tokens: int) -> None:
        new_settings = replace(
            self.settings,
            top_k=int(top_k),
            fetch_k=int(fetch_k),
            max_context_tokens=int(max_context_tokens),
        )
        new_settings.validate()
        self.settings = new_settings

    # -----------------------------
    # Index count
    # -----------------------------
    def _index_count(self) -> int:
        try:
            coll = getattr(self.vectorstore, "_collection", None)
            if coll is None:
                return 0
            return int(coll.count())
        except Exception:
            return 0

    # -----------------------------
    # Retrieval (expects STRING question)
    # -----------------------------
    def _retrieve_with_scores_timed(self, question: str) -> Dict[str, Any]:
        start = time.perf_counter()

        if self._index_count() <= 0:
            raise IndexNotFoundError("Index not found or empty. Please ingest PDFs first.")

        docs_scores: List[Tuple[Document, float]] = self.vectorstore.similarity_search_with_relevance_scores(
            question, k=self.settings.fetch_k
        )

        retrieve_s = time.perf_counter() - start
        return {"docs_scores": docs_scores, "retrieve_s": retrieve_s}

    # -----------------------------
    # Packing (expects dict with question+retrieved)
    # -----------------------------
    def _pack_timed(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()

        packed: PackedContext = pack_context(
            query=payload["question"],
            docs_with_scores=payload["retrieved"]["docs_scores"],
            max_context_tokens=self.settings.max_context_tokens,
            top_k=self.settings.top_k,
        )

        pack_s = time.perf_counter() - start
        return {
            "context": packed.context,
            "selected_docs": packed.selected,
            "debug_rows": packed.debug_rows,
            "pack_s": pack_s,
            "token_budget_used": packed.token_budget_used,
        }

    # -----------------------------
    # Groq call retry
    # -----------------------------
    def _call_llm_with_retry(self, prompt_value) -> str:
        delays = [0.7, 1.5, 3.0]
        last_err: Exception | None = None

        for attempt in range(3):
            try:
                msg = self.llm.invoke(prompt_value.to_messages())
                return getattr(msg, "content", str(msg))
            except Exception as e:
                last_err = e
                logger.warning("Groq call failed (attempt %d/3): %s", attempt + 1, str(e))
                if attempt < 2:
                    time.sleep(delays[attempt])

        logger.exception("Groq call failed after 3 retries.", exc_info=last_err)
        raise GroqCallError("Groq request failed after retries. Please try again.")

    # -----------------------------
    # Build LCEL chain (FIXED)
    # -----------------------------
    def _build_chain(self) -> None:
        """
        LCEL pipeline:
          input: {"question": "..."}
          retrieve gets string via x["question"]
          pack gets dict payload
          prompt uses packed context
        """

        # ✅ IMPORTANT FIX: extract question string before retrieval
        retrieve_r = RunnableLambda(lambda x: self._retrieve_with_scores_timed(x["question"]))

        pack_r = RunnableLambda(self._pack_timed)

        base = RunnablePassthrough.assign(retrieved=retrieve_r).assign(packed=pack_r)

        prompt_vars = RunnableLambda(lambda x: {"question": x["question"], "context": x["packed"]["context"]})

        llm_r = RunnableLambda(lambda pv: self._call_llm_with_retry(pv))

        self.chain = base.assign(answer=(prompt_vars | self.prompt | llm_r | StrOutputParser()))

    # -----------------------------
    # Ask
    # -----------------------------
    def ask(self, question: str) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {
                "answer": "Please enter a question.",
                "latency_s": 0.0,
                "sources": [],
                "debug": {"timings": {}, "selected_scores": [], "config": {}},
            }

        logger.info(
            "Query start | model=%s | TOP_K=%d FETCH_K=%d MAX_CONTEXT_TOKENS=%d",
            self.settings.groq_model,
            self.settings.top_k,
            self.settings.fetch_k,
            self.settings.max_context_tokens,
        )

        total_start = time.perf_counter()
        result = self.chain.invoke({"question": question})
        total_s = time.perf_counter() - total_start

        retrieve_s = float(result["retrieved"]["retrieve_s"])
        pack_s = float(result["packed"]["pack_s"])
        gen_s = max(0.0, total_s - retrieve_s - pack_s)

        selected_docs: List[Document] = result["packed"]["selected_docs"]
        answer = ensure_citations(result["answer"], selected_docs)

        payload = {
            "answer": answer,
            "latency_s": round(total_s, 4),
            "sources": evidence_list(selected_docs),
            "debug": {
                "timings": {
                    "retrieve_s": round(retrieve_s, 4),
                    "pack_s": round(pack_s, 4),
                    "gen_s": round(gen_s, 4),
                    "total_s": round(total_s, 4),
                },
                "selected_scores": result["packed"]["debug_rows"],
                "token_budget_used": result["packed"]["token_budget_used"],
                "config": {
                    "top_k": self.settings.top_k,
                    "fetch_k": self.settings.fetch_k,
                    "max_context_tokens": self.settings.max_context_tokens,
                    "model": self.settings.groq_model,
                    "collection": self.settings.chroma_collection,
                },
            },
        }

        logger.info(
            "Query end | retrieve=%.3fs pack=%.3fs gen=%.3fs total=%.3fs | chunks=%d",
            retrieve_s,
            pack_s,
            gen_s,
            total_s,
            len(selected_docs),
        )
        return payload
