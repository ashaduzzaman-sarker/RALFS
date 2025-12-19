# ============================================================================
# File: tests/test_dense_retriever.py
# ============================================================================
"""Tests for dense retriever."""

import pytest
from pathlib import Path
from ralfs.retriever.dense import DenseRetriever
from ralfs.core.config import RALFSConfig, DataConfig, RetrieverConfig


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = RALFSConfig()
    config.data = DataConfig(dataset="arxiv", split="train", max_samples=20)
    config.retriever = RetrieverConfig(
        dense_model="sentence-transformers/all-MiniLM-L12-v2",
        k_dense=10,
    )
    return config


class TestDenseRetriever:
    """Tests for DenseRetriever."""
    
    def test_retriever_initialization(self, test_config):
        """Test retriever initialization."""
        retriever = DenseRetriever(test_config)
        assert retriever.model is not None
        assert retriever.k_default == 10
    
    @pytest.mark.slow
    def test_load_index(self, test_config):
        """Test loading FAISS index."""
        # First build index
        from ralfs.data.processor import run_preprocessing
        from ralfs.data.indexer import build_index
        
        run_preprocessing(test_config, force_rechunk=True)
        build_index(test_config, force_rebuild=True)
        
        # Load index
        retriever = DenseRetriever(test_config)
        retriever.load_index()
        
        assert retriever.is_initialized()
        assert retriever.index is not None
    
    @pytest.mark.slow
    def test_retrieve(self, test_config):
        """Test dense retrieval."""
        # Setup
        from ralfs.data.processor import run_preprocessing
        from ralfs.data.indexer import build_index
        
        run_preprocessing(test_config, force_rechunk=True)
        build_index(test_config, force_rebuild=True)
        
        # Retrieve
        retriever = DenseRetriever(test_config)
        retriever.load_index()
        results = retriever.retrieve("semiparametric regression", k=5)
        
        assert len(results) <= 5
        assert all(r.score > 0 for r in results)
        assert results[0].rank == 1

