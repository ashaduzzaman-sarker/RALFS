# ============================================================================
# File: tests/test_adaptive_k.py
# ============================================================================
"""Tests for adaptive k selector."""

import pytest
from ralfs.generator.adaptive_k import AdaptiveKSelector


class TestAdaptiveKSelector:
    """Tests for AdaptiveKSelector."""
    
    def test_selector_initialization(self):
        """Test selector initialization."""
        selector = AdaptiveKSelector(min_k=5, max_k=30, default_k=20)
        assert selector.min_k == 5
        assert selector.max_k == 30
        assert selector.default_k == 20
    
    def test_invalid_k_range(self):
        """Test that invalid k range raises error."""
        with pytest.raises(ValueError):
            AdaptiveKSelector(min_k=30, max_k=10)
    
    def test_select_k_with_empty_scores(self):
        """Test k selection with empty scores."""
        selector = AdaptiveKSelector(default_k=20)
        k = selector.select_k([])
        assert k == 20  # Should return default
    
    def test_select_k_with_few_passages(self):
        """Test k selection when fewer passages than min_k."""
        selector = AdaptiveKSelector(min_k=10)
        scores = [0.9, 0.8, 0.7]  # Only 3 passages
        k = selector.select_k(scores)
        assert k == 3  # Should return all available
    
    def test_score_dropoff_strategy(self):
        """Test score dropoff strategy."""
        selector = AdaptiveKSelector(
            min_k=5,
            max_k=30,
            strategy="score_dropoff"
        )
        # Scores with clear drop-off after 8
        scores = [1.0, 0.95, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.40, 0.35, 0.30]
        k = selector.select_k(scores)
        # Should select before the big drop (0.82 -> 0.40)
        assert k == 8
    
    def test_confidence_strategy(self):
        """Test confidence strategy."""
        selector = AdaptiveKSelector(
            min_k=5,
            max_k=30,
            strategy="confidence"
        )
        # All similar scores (low variance)
        scores = [0.90, 0.89, 0.88, 0.87, 0.86] * 5
        k = selector.select_k(scores)
        # Low variance should use more passages
        assert k >= selector.min_k
    
    def test_fixed_strategy(self):
        """Test fixed strategy."""
        selector = AdaptiveKSelector(
            min_k=5,
            max_k=30,
            default_k=15,
            strategy="fixed"
        )
        scores = [1.0] * 25
        k = selector.select_k(scores)
        assert k == 15  # Should always return default
