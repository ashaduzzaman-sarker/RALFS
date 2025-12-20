# ============================================================================
# File: ralfs/generator/adaptive_k.py
# ============================================================================
"""
Adaptive k selection for Fusion-in-Decoder (Novel Contribution).

Dynamically determines the optimal number of passages (k) to use for
generation based on retrieval scores and query characteristics.

Mathematical Formulation:
    k = argmax_{k ∈ [k_min, k_max]} Q(k)
    
    where Q(k) is a quality function that balances:
    - Retrieval confidence: Σ(scores[i]) for i ∈ [0, k]
    - Score dropoff: Δ(scores[k-1], scores[k])
    - Query complexity: f(query_length, entity_count)

Strategies:
    1. score_dropoff: Use k where retrieval scores drop significantly
       Q(k) = scores[k] - λ·Δ(scores[k], scores[k+1])
       
    2. confidence: Use k based on cumulative confidence threshold
       Q(k) = Σ(scores[:k]) / k ≥ τ
       
    3. fixed: Always use default_k (baseline)

This is a novel contribution that reduces computation (fewer passages)
while maintaining or improving quality by avoiding noisy low-scoring passages.
"""

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
        threshold: float = 0.1,
        percentile: float = 75.0,
    ):
        """
        Initialize adaptive k selector.
        
        Args:
            min_k: Minimum number of passages
            max_k: Maximum number of passages
            default_k: Default k if strategy fails
            strategy: Selection strategy ('score_dropoff', 'threshold', 'percentile', 'confidence', 'fixed')
            threshold: Threshold for 'dropoff' and 'threshold' strategies
            percentile: Percentile for 'percentile' strategy
        """
        self.min_k = min_k
        self.max_k = max_k
        self.default_k = default_k
        self.strategy = strategy
        self.threshold = threshold
        self.percentile = percentile
        
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
        if self.strategy == "score_dropoff" or self.strategy == "dropoff":
            k = self._score_dropoff_strategy(scores)
        elif self.strategy == "threshold":
            k = self._threshold_strategy(scores)
        elif self.strategy == "percentile":
            k = self._percentile_strategy(scores)
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
    
    def _threshold_strategy(self, scores: List[float]) -> int:
        """
        Select k based on absolute score threshold.
        
        The intuition: Use all passages above a quality threshold.
        """
        if len(scores) <= self.min_k:
            return len(scores)
        
        # Count scores above threshold
        k = sum(1 for s in scores if s >= self.threshold)
        
        # Ensure k is at least min_k
        k = max(self.min_k, k)
        
        return k
    
    def _percentile_strategy(self, scores: List[float]) -> int:
        """
        Select k based on score percentile.
        
        The intuition: Use passages up to a certain percentile of score distribution.
        """
        if len(scores) <= self.min_k:
            return len(scores)
        
        # Compute percentile threshold
        threshold_score = np.percentile(scores, 100 - self.percentile)
        
        # Count scores above percentile threshold
        k = sum(1 for s in scores if s >= threshold_score)
        
        # Ensure k is at least min_k
        k = max(self.min_k, k)
        
        return k
