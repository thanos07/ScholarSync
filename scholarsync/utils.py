# ----- scholarsync/utils.py -----
"""
Utility functions for ScholarSync.

Includes:
- Custom exception classes (centralized error handling)
- Token approximation
- Citation formatting
- Jaccard similarity for redundancy penalty
- Context packing logic (core anti-hallucination mechanism)
- Evidence formatting helpers
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from langchain_core.documents import Document


# =====================================================
# Custom Exceptions (Centralized Error Handling)
# =====================================================

class ScholarSyncError(Exception):
    """Base exception for ScholarSync."""


class MissingAPIKeyError(ScholarSyncError):
    """Raised when GROQ_API_KEY is missing."""


class NoPDFsFoundError(ScholarSyncError):
    """Raised when no PDFs are found for ingestion."""


class IndexNotFoundError(ScholarSyncError):
    """Raised when Chroma index is missing or empty."""


class EmbeddingInitError(ScholarSyncError):
    """Raised when embedding model fails to initialize."""


class GroqCallError(ScholarSyncError):
    """Raised when Groq API fails after retries."""


# =====================================================
# Token Approximation
# =====================================================

def approx_tokens(text: str) -> int:
    """
    Approximate token count using 4 characters per token heuristic.
    """
    return max(1, len(text) // 4)


# =====================================================
# Citation Formatting
# =====================================================

def format_citation(doc: Document) -> str:
    """
    Format inline citation as [file:page].
    Page is 1-indexed for display.
    """
    source = str(doc.metadata.get("source", "unknown"))
    page = doc.metadata.get("page", None)

    if page is None:
        return f"[{source}:?]"

    try:
        return f"[{source}:{int(page) + 1}]"
    except Exception:
        return f"[{source}:?]"


# =====================================================
# Similarity / Redundancy Logic
# =====================================================

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _word_set(text: str) -> set[str]:
    """Extract normalized word set."""
    return set(w.lower() for w in _WORD_RE.findall(text or ""))


def jaccard_similarity(a: str, b: str) -> float:
    """
    Compute Jaccard similarity between two strings.
    Used to penalize redundant chunks.
    """
    sa, sb = _word_set(a), _word_set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _query_overlap_score(query: str, text: str) -> float:
    """
    Boost chunks that share words with query.
    """
    q = _word_set(query)
    t = _word_set(text)
    if not q or not t:
        return 0.0
    return len(q & t) / max(1, len(q))


def _collapse_ws(s: str) -> str:
    """Normalize whitespace."""
    return re.sub(r"\s+", " ", (s or "")).strip()


# =====================================================
# Packed Context Data Structure
# =====================================================

@dataclass
class PackedContext:
    context: str
    selected: List[Document]
    debug_rows: List[Dict[str, Any]]
    token_budget_used: int


# =====================================================
# Context Packing (Core Anti-Hallucination Mechanism)
# =====================================================

def pack_context(
    query: str,
    docs_with_scores: Sequence[Tuple[Document, float]],
    max_context_tokens: int,
    top_k: int,
    min_effective_score: float = 0.15,
) -> PackedContext:
    """
    Selects best chunks under token budget.

    Steps:
    - Combine vector score + query overlap
    - Penalize redundancy (Jaccard similarity)
    - Penalize overly long chunks slightly
    - Skip low effective-score chunks
    - Stop when top_k or token budget reached
    """

    # Combine retrieval score + query overlap boost
    candidates: List[Tuple[Document, float, float]] = []

    for doc, score in docs_with_scores:
        q_overlap = _query_overlap_score(query, doc.page_content)
        combined = (0.75 * float(score)) + (0.25 * q_overlap)
        candidates.append((doc, float(score), combined))

    # Sort descending by combined score
    candidates.sort(key=lambda x: x[2], reverse=True)

    selected: List[Document] = []
    debug_rows: List[Dict[str, Any]] = []
    token_budget_used = 0

    for doc, raw_score, combined_score in candidates:

        if len(selected) >= top_k:
            break

        text = _collapse_ws(doc.page_content)
        if not text:
            continue

        tokens = approx_tokens(text)

        # Skip if exceeding budget (unless first chunk)
        if selected and (token_budget_used + tokens) > max_context_tokens:
            continue

        # If first chunk too large, truncate to fit
        if not selected and tokens > max_context_tokens:
            max_chars = max_context_tokens * 4
            text = text[:max_chars]
            tokens = approx_tokens(text)

        # Compute redundancy penalty
        max_sim = 0.0
        for sdoc in selected:
            sim = jaccard_similarity(text, sdoc.page_content)
            max_sim = max(max_sim, sim)

        redundancy_penalty = 0.35 * max_sim

        # Slight length penalty
        length_penalty = 0.02 * (tokens / max(1, max_context_tokens)) ** 0.5

        effective_score = combined_score - redundancy_penalty - length_penalty

        if effective_score < min_effective_score:
            continue

        new_doc = Document(page_content=text, metadata=dict(doc.metadata))
        selected.append(new_doc)
        token_budget_used += tokens

        debug_rows.append(
            {
                "rank": len(selected),
                "source": new_doc.metadata.get("source"),
                "page": (
                    int(new_doc.metadata.get("page", 0)) + 1
                    if str(new_doc.metadata.get("page", "")).isdigit()
                    else new_doc.metadata.get("page")
                ),
                "raw_score": round(raw_score, 4),
                "combined_score": round(combined_score, 4),
                "redundancy_max_jaccard": round(max_sim, 4),
                "redundancy_penalty": round(redundancy_penalty, 4),
                "length_penalty": round(length_penalty, 4),
                "effective_score": round(effective_score, 4),
                "tokens": tokens,
                "budget_used": token_budget_used,
                "chunk_id": new_doc.metadata.get("chunk_id"),
            }
        )

    # Build final context string with inline citations prepended
    parts: List[str] = []
    for doc in selected:
        parts.append(f"{format_citation(doc)} {doc.page_content}")

    context = "\n\n".join(parts).strip()

    return PackedContext(
        context=context,
        selected=selected,
        debug_rows=debug_rows,
        token_budget_used=token_budget_used,
    )


# =====================================================
# Citation Enforcement
# =====================================================

_CITATION_RE = re.compile(r"\[[^\[\]]+:\d+\]")


def ensure_citations(answer: str, evidence: Sequence[Document]) -> str:
    """
    If model forgot inline citations, append a fallback source list.
    """
    if _CITATION_RE.search(answer or ""):
        return answer

    unique_citations = []
    seen = set()

    for doc in evidence:
        citation = format_citation(doc)
        if citation not in seen:
            unique_citations.append(citation)
            seen.add(citation)

    if not unique_citations:
        return answer

    return (answer.rstrip() + "\n\nSources: " + ", ".join(unique_citations)).strip()


# =====================================================
# Evidence List Formatter
# =====================================================

def evidence_list(docs: Sequence[Document], snippet_chars: int = 260) -> List[Dict[str, Any]]:
    """
    Convert selected documents into structured evidence objects for UI display.
    """
    output: List[Dict[str, Any]] = []

    for doc in docs:
        source = str(doc.metadata.get("source", "unknown"))
        page0 = doc.metadata.get("page", None)

        try:
            page1 = int(page0) + 1 if page0 is not None else None
        except Exception:
            page1 = None

        snippet = _collapse_ws(doc.page_content)[:snippet_chars]

        output.append(
            {
                "source": source,
                "page": page1,
                "snippet": snippet,
                "chunk_id": doc.metadata.get("chunk_id"),
            }
        )

    return output
