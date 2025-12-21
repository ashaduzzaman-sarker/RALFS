# ============================================================================
# File: ralfs/retriever/__init__.py
# ============================================================================
"""Retrieval module for RALFS: Dense, Sparse, ColBERT, Hybrid, and Reranking."""

from .base import BaseRetriever, RetrievalResult
from .colbert import ColBERTRetriever
from .dense import DenseRetriever
from .factory import create_retriever
from .hybrid import HybridRetriever
from .reranker import CrossEncoderReranker
from .sparse import SparseRetriever
from .utils import normalize_scores, reciprocal_rank_fusion

__all__ = [
    # Base classes
    "BaseRetriever",
    "RetrievalResult",
    # Retrievers
    "DenseRetriever",
    "SparseRetriever",
    "ColBERTRetriever",
    "HybridRetriever",
    "CrossEncoderReranker",
    # Factory
    "create_retriever",
    # Utils
    "reciprocal_rank_fusion",
    "normalize_scores",
]
