# ----- scholarsync/prompts.py -----
"""
Prompt templates for ScholarSync RAG.

Goals:
- Strict grounding: use ONLY provided context.
- Explicit fallback sentence if answer is not supported.
- Enforce inline citations in the format: [file:page]
- Reduce hallucinations via clear system instructions.

This module returns LangChain ChatPromptTemplate objects
compatible with LCEL pipelines.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


# -------------------------------------------------
# Strict RAG Prompt
# -------------------------------------------------
def strict_rag_prompt() -> ChatPromptTemplate:
    """
    Create a strict RAG prompt template that:
    - Forces the model to use ONLY provided context.
    - Requires inline citations [file:page].
    - Enforces exact fallback phrase when answer is missing.

    Fallback phrase (must match spec exactly):
        I don't know based on the documents.
    """

    system_message = (
        "You are ScholarSync, a precise and reliable assistant for answering "
        "questions using a set of PDF documents.\n\n"
        "RULES (STRICT):\n"
        "1) Use ONLY the provided context.\n"
        "2) Do NOT use outside knowledge.\n"
        "3) If the answer is not fully supported by the context, respond EXACTLY with:\n"
        "   I don't know based on the documents.\n"
        "4) Every factual statement MUST include an inline citation in the format [file:page].\n"
        "5) Keep answers concise, structured, and grounded in evidence.\n"
    )

    user_message = (
        "Context (evidence excerpts):\n"
        "----------------------------------------\n"
        "{context}\n"
        "----------------------------------------\n\n"
        "Question:\n"
        "{question}\n\n"
        "Provide a clear, grounded answer using ONLY the context above. "
        "Include inline citations in the format [file:page]."
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", user_message),
        ]
    )
