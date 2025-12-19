# ============================================================================
# File: ralfs/retriever/factory.py
# ============================================================================
"""Factory for creating retrievers."""

from typing import Union
from omegaconf import DictConfig

from ralfs.retriever.base import BaseRetriever
from ralfs.retriever.dense import DenseRetriever
from ralfs.retriever.sparse import SparseRetriever
from ralfs.retriever.colbert import ColBERTRetriever
from ralfs.retriever.hybrid import HybridRetriever
from ralfs.core.logging import get_logger

logger = get_logger(__name__)


def create_retriever(
    cfg: DictConfig,
    retriever_type: Union[str, None] = None,
) -> BaseRetriever:
    """
    Factory function to create retriever based on configuration.
    
    Args:
        cfg: Configuration object
        retriever_type: Type of retriever ('dense', 'sparse', 'colbert', 'hybrid')
                       If None, uses cfg.retriever.type
    
    Returns:
        Initialized retriever instance
    
    Example:
        >>> from ralfs.core import load_config
        >>> from ralfs.retriever import create_retriever
        >>> 
        >>> config = load_config("configs/ralfs.yaml")
        >>> retriever = create_retriever(config)
        >>> retriever.load_index()
        >>> results = retriever.retrieve("quantum computing", k=10)
    """
    # Get retriever type from config or argument
    if retriever_type is None:
        retriever_type = getattr(cfg.retriever, 'type', 'hybrid')
    
    retriever_type = retriever_type.lower()
    
    logger.info(f"Creating {retriever_type} retriever...")
    
    # Create retriever
    if retriever_type == 'dense':
        retriever = DenseRetriever(cfg)
    elif retriever_type == 'sparse':
        retriever = SparseRetriever(cfg)
    elif retriever_type == 'colbert':
        retriever = ColBERTRetriever(cfg)
    elif retriever_type == 'hybrid':
        retriever = HybridRetriever(cfg)
    else:
        raise ValueError(
            f"Unknown retriever type: {retriever_type}. "
            f"Choose from: dense, sparse, colbert, hybrid"
        )
    
    logger.info(f"✓ {retriever_type.capitalize()} retriever created")
    return retriever