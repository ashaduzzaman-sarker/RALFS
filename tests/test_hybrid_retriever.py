# ============================================================================
# File: tests/test_hybrid_retriever.py
# ============================================================================
"""Tests for hybrid retriever."""

import pytest
from ralfs.retriever.hybrid import HybridRetriever
from ralfs.core.config import RALFSConfig, DataConfig, RetrieverConfig


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = RALFSConfig()
    config.data = DataConfig(dataset="arxiv", split="train", max_samples=20)
    config.retriever = RetrieverConfig(
        type="hybrid",
        dense_model="sentence-transformers/all-MiniLM-L12-v2",
        k_dense=20,
        k_sparse=20,
        k_final=10,
    )
    return config


class TestHybridRetriever:
    """Tests for HybridRetriever."""
    
    def test_retriever_initialization(self, test_config):
        """Test hybrid retriever initialization."""
        retriever = HybridRetriever(test_config)
        
        assert retriever.dense is not None
        assert retriever.sparse is not None
        assert retriever.reranker is not None
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_load_index(self, test_config):
        """Test loading all indexes."""
        # Setup
        from ralfs.data.processor import run_preprocessing
        from ralfs.data.indexer import build_index
        
        run_preprocessing(test_config, force_rechunk=True)
        build_index(test_config, force_rebuild=True)
        
        # Load
        retriever = HybridRetriever(test_config)
        retriever.load_index()
        
        assert retriever.is_initialized()
        assert retriever.dense.is_initialized()
        assert retriever.sparse.is_initialized()
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_hybrid_retrieve(self, test_config):
        """Test full hybrid retrieval pipeline."""
        # Setup
        from ralfs.data.processor import run_preprocessing
        from ralfs.data.indexer import build_index
        
        run_preprocessing(test_config, force_rechunk=True)
        build_index(test_config, force_rebuild=True)
        
        # Retrieve
        retriever = HybridRetriever(test_config)
        retriever.load_index()
        results = retriever.retrieve("semiparametric regression", k=5)

        # # ADD THESE DEBUG LINES
        # print(f"\n🔍 DEBUG: Got {len(results)} results")
        # for i, r in enumerate(results[:3]):
        #     print(f"  Result {i}: score={r.score}, metadata={r.metadata}")
        
        assert len(results) <= 5
        assert all(r.score >= 0 for r in results)  # ← THIS IS PROBABLY FAILING
        assert results[0].rank == 1        
        # Check that reranking happened
        if results:
            assert results[0].metadata.get("reranked") is True
