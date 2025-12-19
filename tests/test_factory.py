# ============================================================================
# File: tests/test_factory.py
# ============================================================================
"""Tests for retriever factory."""

import pytest
from ralfs.retriever.factory import create_retriever
from ralfs.retriever.dense import DenseRetriever
from ralfs.retriever.sparse import SparseRetriever
from ralfs.retriever.hybrid import HybridRetriever
from ralfs.core.config import RALFSConfig, RetrieverConfig


@pytest.fixture
def base_config():
    """Create base configuration."""
    config = RALFSConfig()
    config.retriever = RetrieverConfig()
    return config


class TestRetrieverFactory:
    """Tests for retriever factory."""
    
    def test_create_dense(self, base_config):
        """Test creating dense retriever."""
        retriever = create_retriever(base_config, retriever_type="dense")
        assert isinstance(retriever, DenseRetriever)
    
    def test_create_sparse(self, base_config):
        """Test creating sparse retriever."""
        retriever = create_retriever(base_config, retriever_type="sparse")
        assert isinstance(retriever, SparseRetriever)
    
    def test_create_hybrid(self, base_config):
        """Test creating hybrid retriever."""
        retriever = create_retriever(base_config, retriever_type="hybrid")
        assert isinstance(retriever, HybridRetriever)
    
    def test_create_from_config(self, base_config):
        """Test creating retriever from config type."""
        base_config.retriever.type = "hybrid"
        retriever = create_retriever(base_config)
        assert isinstance(retriever, HybridRetriever)
    
    def test_invalid_type(self, base_config):
        """Test creating retriever with invalid type."""
        with pytest.raises(ValueError, match="Unknown retriever type"):
            create_retriever(base_config, retriever_type="invalid")