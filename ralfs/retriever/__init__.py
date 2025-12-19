# ============================================================================
# File: ralfs/retriever/__init__.py
# ============================================================================
"""Retrieval module for RALFS: Dense, Sparse, ColBERT, Hybrid, and Reranking."""

from .base import BaseRetriever, RetrievalResult
from .dense import DenseRetriever
from .sparse import SparseRetriever
from .colbert import ColBERTRetriever
from .hybrid import HybridRetriever
from .reranker import CrossEncoderReranker
from .factory import create_retriever
from .utils import reciprocal_rank_fusion, normalize_scores

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