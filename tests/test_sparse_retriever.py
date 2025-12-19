# ============================================================================
# File: tests/test_sparse_retriever.py
# ============================================================================
"""Tests for sparse retriever."""

import pytest
from ralfs.retriever.sparse import SparseRetriever
from ralfs.core.config import RALFSConfig, DataConfig, RetrieverConfig


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = RALFSConfig()
    config.data = DataConfig(dataset="arxiv", split="train", max_samples=5)
    config.retriever = RetrieverConfig(k_sparse=10)
    return config


class TestSparseRetriever:
    """Tests for SparseRetriever."""
    
    def test_retriever_initialization(self, test_config):
        """Test retriever initialization."""
        retriever = SparseRetriever(test_config)
        assert retriever.k_default == 10
        assert retriever.k1 == 1.5
        assert retriever.b == 0.75
    
    def test_build_index_from_texts(self, test_config):
        """Test building BM25 index from texts."""
        texts = ["This is document one.", "This is document two."]
        retriever = SparseRetriever(test_config, chunks=texts)
        
        assert retriever.is_initialized()
        assert retriever.bm25 is not None
    
    @pytest.mark.slow
    def test_load_index(self, test_config):
        """Test loading chunks and building BM25."""
        from ralfs.data.processor import run_preprocessing
        
        run_preprocessing(test_config, force_rechunk=True)
        
        retriever = SparseRetriever(test_config)
        retriever.load_index()
        
        assert retriever.is_initialized()
        assert retriever.bm25 is not None
    
    @pytest.mark.slow
    def test_retrieve(self, test_config):
        """Test sparse retrieval."""
        from ralfs.data.processor import run_preprocessing
        
        run_preprocessing(test_config, force_rechunk=True)
        
        retriever = SparseRetriever(test_config)
        retriever.load_index()
        results = retriever.retrieve("quantum physics", k=5)
        
        assert len(results) <= 5
        assert all(r.score >= 0 for r in results)

