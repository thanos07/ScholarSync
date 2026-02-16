# ----- scholarsync/logger.py -----
"""
Centralized logging configuration for ScholarSync.

Features:
- RotatingFileHandler (5MB per file, keep 5 backups)
- Console handler
- LOG_LEVEL environment variable support (default: INFO)
- Auto-create logs/ directory
- Safe logger caching (avoid duplicate handlers)

Log file:
    logs/scholarsync.log
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# Cache to prevent duplicate logger configuration
_LOGGER_CACHE: dict[str, logging.Logger] = {}


# -----------------------------
# Helper: Parse LOG_LEVEL safely
# -----------------------------
def _parse_log_level(level_str: str) -> int:
    """
    Convert string log level (e.g., "INFO") to logging constant.
    Defaults to INFO if invalid.
    """
    if not level_str:
        return logging.INFO

    level_str = level_str.strip().upper()
    return getattr(logging, level_str, logging.INFO)


# -----------------------------
# Public Logger Factory
# -----------------------------
def get_logger(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """
    Create (or retrieve cached) logger configured with:
        - Rotating file handler
        - Console handler
        - LOG_LEVEL support
    """

    # Return cached logger if already configured
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    # Determine log level from environment
    level = _parse_log_level(os.getenv("LOG_LEVEL", "INFO"))

    # Ensure logs directory exists
    base_dir = Path(log_dir) if log_dir else Path("logs")
    base_dir.mkdir(parents=True, exist_ok=True)

    log_path = base_dir / "scholarsync.log"

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate logs in root logger

    # Avoid adding handlers multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Rotating file handler (5MB, keep 5 backups)
        file_handler = RotatingFileHandler(
            filename=log_path,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    # Cache logger instance
    _LOGGER_CACHE[name] = logger
    return logger
