# ============================================================================
# File: ralfs/generator/__init__.py
# ============================================================================
"""Generation module with Fusion-in-Decoder and adaptive k selection."""

from .base import BaseGenerator, GenerationResult
from .fid import FiDGenerator
from .adaptive_k import AdaptiveKSelector
from .factory import create_generator

__all__ = [
    "BaseGenerator",
    "GenerationResult",
    "FiDGenerator",
    "AdaptiveKSelector",
    "create_generator",
]