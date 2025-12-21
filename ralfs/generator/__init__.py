# ============================================================================
# File: ralfs/generator/__init__.py
# ============================================================================
"""Generation module with Fusion-in-Decoder and adaptive k selection."""

from .adaptive_k import AdaptiveKSelector
from .base import BaseGenerator, GenerationResult
from .factory import create_generator
from .fid import FiDGenerator

__all__ = [
    "BaseGenerator",
    "GenerationResult",
    "FiDGenerator",
    "AdaptiveKSelector",
    "create_generator",
]
