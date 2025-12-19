# ============================================================================
# File: tests/test_reranker.py
# ============================================================================
"""Tests for cross-encoder reranker."""

import pytest
from ralfs.retriever.reranker import CrossEncoderReranker
from ralfs.retriever.base import RetrievalResult
from ralfs.core.config import RALFSConfig, RetrieverConfig


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = RALFSConfig()
    config.retriever = RetrieverConfig(
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",  # Smaller model
    )
    return config


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""
    
    def test_reranker_initialization(self, test_config):
        """Test reranker initialization."""
        reranker = CrossEncoderReranker(test_config)
        assert reranker.model is not None
        assert reranker.enabled is True
    
    def test_reranker_disabled(self):
        """Test reranker when disabled."""
        config = RALFSConfig()
        config.retriever = RetrieverConfig()
        config.retriever.reranker = {"enabled": False}
        
        reranker = CrossEncoderReranker(config)
        assert reranker.enabled is False
    
    def test_rerank_results(self, test_config):
        """Test reranking results."""
        reranker = CrossEncoderReranker(test_config)
        
        # Create test candidates
        candidates = [
            RetrievalResult(text="The cat sat on the mat", score=0.5, rank=1),
            RetrievalResult(text="Dogs are great pets", score=0.6, rank=2),
            RetrievalResult(text="Cats and dogs live together", score=0.4, rank=3),
        ]
        
        # Rerank
        query = "cats and pets"
        reranked = reranker.rerank(query, candidates, top_k=2)
        
        assert len(reranked) == 2
        assert reranked[0].rank == 1
        assert reranked[0].metadata["reranked"] is True
    
    def test_rerank_dicts(self, test_config):
        """Test reranking dict candidates."""
        reranker = CrossEncoderReranker(test_config)
        
        candidates = [
            {"text": "Machine learning is great", "score": 0.5},
            {"text": "Deep learning uses neural networks", "score": 0.6},
        ]
        
        query = "neural networks"
        reranked = reranker.rerank(query, candidates, top_k=2)
        
        assert len(reranked) == 2
        assert all(isinstance(r, RetrievalResult) for r in reranked)

