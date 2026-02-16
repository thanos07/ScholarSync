# ----- scholarsync/config.py -----
"""
Configuration management for ScholarSync.

Responsibilities:
- Read environment variables (from OS or .env via python-dotenv).
- Provide strongly-typed Settings dataclass.
- Validate numeric constraints and cross-field rules.
- Log successful config load or detailed validation errors.

Environment variables (see .env.example):
    GROQ_API_KEY
    GROQ_MODEL
    EMBEDDING_MODEL
    CHROMA_PERSIST_DIR
    CHROMA_COLLECTION
    TOP_K
    FETCH_K
    MAX_CONTEXT_TOKENS
    LOG_LEVEL
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Dict, Any

from .logger import get_logger

logger = get_logger("scholarsync.config")


# -----------------------------
# Helper: Parse integer env var
# -----------------------------
def _get_env_int(key: str, default: int) -> int:
    """
    Safely parse an integer environment variable.
    Raises ValueError with friendly message if invalid.
    """
    raw = os.getenv(key, str(default)).strip()
    try:
        return int(raw)
    except Exception as e:
        raise ValueError(f"{key} must be an integer (got '{raw}').") from e


# -----------------------------
# Settings Dataclass
# -----------------------------
@dataclass(frozen=True)
class Settings:
    """
    Central configuration object for ScholarSync.
    """

    # ---- LLM ----
    groq_api_key: str
    groq_model: str

    # ---- Embeddings ----
    embedding_model: str

    # ---- Chroma ----
    chroma_persist_dir: str
    chroma_collection: str

    # ---- Retrieval knobs ----
    top_k: int
    fetch_k: int
    max_context_tokens: int

    # ---- Logging ----
    log_level: str = "INFO"

    # -----------------------------
    # Factory: Load from environment
    # -----------------------------
    @staticmethod
    def from_env(overrides: Optional[Dict[str, Any]] = None) -> "Settings":
        """
        Create Settings from environment variables.
        Optional overrides allow runtime knob changes (UI).
        """
        try:
            settings = Settings(
                groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
                groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip(),
                embedding_model=os.getenv(
                    "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
                ).strip(),
                chroma_persist_dir=os.getenv(
                    "CHROMA_PERSIST_DIR", "data/chroma"
                ).strip(),
                chroma_collection=os.getenv(
                    "CHROMA_COLLECTION", "scholarsync"
                ).strip(),
                top_k=_get_env_int("TOP_K", 5),
                fetch_k=_get_env_int("FETCH_K", 20),
                max_context_tokens=_get_env_int("MAX_CONTEXT_TOKENS", 2800),
                log_level=os.getenv("LOG_LEVEL", "INFO").strip() or "INFO",
            )

            # Apply runtime overrides (e.g., from UI sidebar)
            if overrides:
                settings = replace(settings, **overrides)

            # Validate all values
            settings.validate()

            # Ensure log and persist directories exist
            Path("logs").mkdir(parents=True, exist_ok=True)
            Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

            logger.info(
                "Settings loaded successfully | collection=%s | persist_dir=%s | model=%s",
                settings.chroma_collection,
                settings.chroma_persist_dir,
                settings.groq_model,
            )

            return settings

        except Exception:
            logger.exception("Failed to load or validate settings.")
            raise

    # -----------------------------
    # Validation Logic
    # -----------------------------
    def validate(self) -> None:
        """
        Validate configuration values.
        Raises ValueError with friendly message on failure.
        """

        def require_positive_int(name: str, value: int) -> None:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer (got {value}).")

        # Validate numeric parameters
        require_positive_int("TOP_K", self.top_k)
        require_positive_int("FETCH_K", self.fetch_k)
        require_positive_int("MAX_CONTEXT_TOKENS", self.max_context_tokens)

        # Cross-field constraint
        if self.fetch_k < self.top_k:
            raise ValueError(
                f"FETCH_K must be >= TOP_K "
                f"(got FETCH_K={self.fetch_k}, TOP_K={self.top_k})."
            )

        # Validate Chroma settings
        if not self.chroma_persist_dir:
            raise ValueError("CHROMA_PERSIST_DIR must not be empty.")

        if not self.chroma_collection:
            raise ValueError("CHROMA_COLLECTION must not be empty.")

        # GROQ_API_KEY is intentionally NOT enforced here,
        # because ingestion does not require it.
        # It is validated at query time in rag.py.

    # -----------------------------
    # Utility: Safe dict export
    # -----------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Return settings as dictionary (excluding sensitive keys).
        Useful for debug panels.
        """
        return {
            "groq_model": self.groq_model,
            "embedding_model": self.embedding_model,
            "chroma_persist_dir": self.chroma_persist_dir,
            "chroma_collection": self.chroma_collection,
            "top_k": self.top_k,
            "fetch_k": self.fetch_k,
            "max_context_tokens": self.max_context_tokens,
            "log_level": self.log_level,
        }
