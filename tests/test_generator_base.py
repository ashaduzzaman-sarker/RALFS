# ============================================================================
# File: tests/test_generator_base.py
# ============================================================================
"""Tests for base generator classes."""

import pytest
from ralfs.generator.base import GenerationResult, BaseGenerator


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a generation result."""
        result = GenerationResult(
            summary="This is a test summary.",
            query="test query",
            k_used=10,
            num_passages=20,
        )
        assert result.summary == "This is a test summary."
        assert result.k_used == 10
        assert result.num_passages == 20
    
    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = GenerationResult(
            summary="Test",
            query="query",
            k_used=5,
            num_passages=10,
            metadata={"key": "value"},
        )
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["metadata"]["key"] == "value"
    
    def test_result_repr(self):
        """Test result string representation."""
        result = GenerationResult(
            summary="A" * 200,  # Long summary
            query="test",
            k_used=10,
            num_passages=20,
        )
        repr_str = repr(result)
        assert "k=10/20" in repr_str
