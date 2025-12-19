# ============================================================================
# File: tests/test_utils.py
# ============================================================================
"""Tests for retrieval utilities."""

import pytest
from ralfs.retriever.utils import normalize_scores, reciprocal_rank_fusion
from ralfs.retriever.base import RetrievalResult


class TestNormalizeScores:
    """Tests for score normalization."""
    
    def test_normalize_basic(self):
        """Test basic normalization."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = normalize_scores(scores)
        
        assert normalized[0] == 0.0  # Min
        # assert normalized[-1] == 1.0  # Max
        assert abs(normalized[-1] - 1.0) < 1e-6  # Max (with tolerance)
        
        assert all(0 <= s <= 1 for s in normalized)
    
    def test_normalize_same_scores(self):
        """Test normalization when all scores are same."""
        scores = [5.0, 5.0, 5.0]
        normalized = normalize_scores(scores)
        
        assert all(s == 1.0 for s in normalized)
    
    def test_normalize_empty(self):
        """Test normalization with empty list."""
        normalized = normalize_scores([])
        assert normalized == []


class TestReciprocalRankFusion:
    """Tests for RRF fusion."""
    
    def test_rrf_basic(self):
        """Test basic RRF fusion."""
        results1 = [
            RetrievalResult(text="Doc A", score=0.9, rank=1),
            RetrievalResult(text="Doc B", score=0.8, rank=2),
        ]
        results2 = [
            RetrievalResult(text="Doc B", score=0.95, rank=1),
            RetrievalResult(text="Doc C", score=0.7, rank=2),
        ]
        
        fused = reciprocal_rank_fusion([results1, results2], k=60)
        
        # Doc B should rank higher (appears in both lists)
        assert fused[0].text == "Doc B"
        assert all(r.metadata.get("fusion") == "rrf" for r in fused)
    
    def test_rrf_with_dicts(self):
        """Test RRF with dict results."""
        results1 = [
            {"text": "Doc A", "score": 0.9},
            {"text": "Doc B", "score": 0.8},
        ]
        results2 = [
            {"text": "Doc B", "score": 0.95},
            {"text": "Doc C", "score": 0.7},
        ]
        
        fused = reciprocal_rank_fusion([results1, results2])
        
        assert len(fused) == 3
        assert all(isinstance(r, RetrievalResult) for r in fused)
    
    def test_rrf_empty(self):
        """Test RRF with empty input."""
        fused = reciprocal_rank_fusion([])
        assert fused == []

