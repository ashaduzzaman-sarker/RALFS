# ============================================================================
# File: ralfs/core/__init__.py
# ============================================================================
"""Core configuration and utilities for RALFS."""

from .config import RALFSConfig, load_config
from .constants import (
    CHECKPOINTS_DIR,
    DATA_DIR,
    DEFAULT_COLBERT_MODEL,
    DEFAULT_DENSE_MODEL,
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_K_COLBERT,
    DEFAULT_K_DENSE,
    DEFAULT_K_FINAL,
    DEFAULT_K_SPARSE,
    DEFAULT_RERANKER_MODEL,
    INDEX_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    RESULTS_DIR,
    ROOT_DIR,
)
from .logging import get_logger, setup_logging

__all__ = [
    "RALFSConfig",
    "load_config",
    "get_logger",
    "setup_logging",
    "ROOT_DIR",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "INDEX_DIR",
    "CHECKPOINTS_DIR",
    "RESULTS_DIR",
    "DEFAULT_DENSE_MODEL",
    "DEFAULT_COLBERT_MODEL",
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_GENERATOR_MODEL",
    "DEFAULT_K_DENSE",
    "DEFAULT_K_SPARSE",
    "DEFAULT_K_COLBERT",
    "DEFAULT_K_FINAL",
]
