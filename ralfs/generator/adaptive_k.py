# ============================================================================
# File: ralfs/generator/adaptive_k.py
# ============================================================================
"""Adaptive k selection strategies for FiD."""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from ralfs.core.logging import get_logger

logger = get_logger(__name__)


class AdaptiveKSelector:
    """
    Adaptive k selection for FiD generation (Novel contribution).
    
    Strategies:
    - score_dropoff: Select k based on score drop-off
    - confidence: Select k based on score variance
    - fixed: Use fixed k (baseline)
    """
    
    def __init__(
        self,
        min_k: int = 5,
        max_k: int = 30,
        default_k: int = 20,
        strategy: str = "score_dropoff",
    ):
        """
        Initialize adaptive k selector.
        
        Args:
            min_k: Minimum number of passages
            max_k: Maximum number of passages
            default_k: Default k if strategy fails
            strategy: Selection strategy ('score_dropoff', 'confidence', 'fixed')
        """
        self.min_k = min_k
        self.max_k = max_k
        self.default_k = default_k
        self.strategy = strategy
        
        if not (1 <= min_k <= max_k):
            raise ValueError(f"Invalid k range: min_k={min_k}, max_k={max_k}")
        
        logger.info(
            f"AdaptiveKSelector initialized: strategy={strategy}, "
            f"range=[{min_k}, {max_k}], default={default_k}"
        )
    
    def select_k(self, scores: List[float]) -> int:
        """
        Select optimal k based on retrieval scores.
        
        Args:
            scores: List of relevance scores (sorted descending)
        
        Returns:
            Optimal k value
        """
        if not scores:
            logger.warning("Empty scores, using default k")
            return self.default_k
        
        # Limit to available passages
        n_available = len(scores)
        if n_available <= self.min_k:
            return n_available
        
        # Apply strategy
        if self.strategy == "score_dropoff":
            k = self._score_dropoff_strategy(scores)
        elif self.strategy == "confidence":
            k = self._confidence_strategy(scores)
        elif self.strategy == "fixed":
            k = self.default_k
        else:
            logger.warning(f"Unknown strategy '{self.strategy}', using default k")
            k = self.default_k
        
        # Ensure k is within bounds
        k = max(self.min_k, min(k, self.max_k, n_available))
        
        logger.debug(f"Selected k={k} using strategy='{self.strategy}' from {n_available} passages")
        return k
    
    def _score_dropoff_strategy(self, scores: List[float]) -> int:
        """
        Select k based on score drop-off (largest gap).
        
        The intuition: Stop before a large drop in relevance scores,
        indicating a quality boundary.
        """
        if len(scores) < 2:
            return len(scores)
        
        # Compute score differences (drop-off)
        dropoffs = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
        
        # Find largest drop-off after min_k
        valid_range = range(self.min_k - 1, min(self.max_k - 1, len(dropoffs)))
        
        if not valid_range:
            return self.min_k
        
        # Find position of largest drop
        max_dropoff_idx = max(valid_range, key=lambda i: dropoffs[i])
        
        # k is index + 1 (before the drop)
        k = max_dropoff_idx + 1
        
        return k
    
    def _confidence_strategy(self, scores: List[float]) -> int:
        """
        Select k based on score variance (confidence).
        
        The intuition: If top scores are very similar (low variance),
        use more passages. If top scores vary a lot, use fewer.
        """
        if len(scores) <= self.min_k:
            return len(scores)
        
        # Compute rolling variance in windows
        window_size = self.min_k
        variances = []
        
        for i in range(len(scores) - window_size + 1):
            window = scores[i:i + window_size]
            var = np.var(window)
            variances.append(var)
        
        # High variance → use fewer passages (top ones are clearly better)
        # Low variance → use more passages (all similar quality)
        if variances:
            avg_variance = np.mean(variances)
            
            # Normalize variance to k range
            # Higher variance → closer to min_k
            # Lower variance → closer to max_k
            if avg_variance > 0.1:  # High variance
                k = self.min_k + int((self.max_k - self.min_k) * 0.3)
            elif avg_variance < 0.01:  # Low variance
                k = self.min_k + int((self.max_k - self.min_k) * 0.8)
            else:  # Medium variance
                k = self.default_k
        else:
            k = self.default_k
        
        return k
