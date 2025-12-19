# ============================================================================
# File: tests/test_retriever_base.py
# ============================================================================
"""Tests for base retriever classes."""

import pytest
from ralfs.retriever.base import RetrievalResult, BaseRetriever


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a retrieval result."""
        result = RetrievalResult(
            text="Test text",
            score=0.95,
            rank=1,
            doc_id="doc_1",
            chunk_id="doc_1_c0",
        )
        assert result.text == "Test text"
        assert result.score == 0.95
        assert result.rank == 1
    
    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = RetrievalResult(
            text="Test",
            score=0.9,
            rank=1,
            metadata={"key": "value"},
        )
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["metadata"]["key"] == "value"
    
    def test_result_repr(self):
        """Test result string representation."""
        result = RetrievalResult(
            text="A" * 100,  # Long text
            score=0.85,
            rank=2,
        )
        repr_str = repr(result)
        assert "rank=2" in repr_str
        assert "score=0.85" in repr_str