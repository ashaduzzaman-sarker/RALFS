# ============================================================================
# File: ralfs/generator/base.py
# ============================================================================
"""Base generator class and result structure."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    """
    Generation result with metadata.

    Attributes:
        summary: Generated summary text
        query: Original query
        k_used: Number of passages used
        num_passages: Total passages available
        adaptive_strategy: Strategy used for k selection
        metadata: Additional metadata
    """

    summary: str
    query: str
    k_used: int
    num_passages: int
    adaptive_strategy: str | None = None
    generation_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def __repr__(self) -> str:
        preview = self.summary[:100] + "..." if len(self.summary) > 100 else self.summary
        return f"GenerationResult(k={self.k_used}/{self.num_passages}, summary='{preview}')"


class BaseGenerator(ABC):
    """
    Abstract base class for all generators.

    All generators must implement:
    - generate(): Generate summary from query and passages
    """

    def __init__(self, cfg):
        """
        Initialize generator with configuration.

        Args:
            cfg: Configuration object (RALFSConfig)
        """
        self.cfg = cfg
        self._initialized = False

    @abstractmethod
    def generate(
        self,
        query: str,
        passages: list[dict[str, Any]],
    ) -> GenerationResult:
        """
        Generate summary from query and retrieved passages.

        Args:
            query: Input query/question
            passages: Retrieved passages with 'text' and 'score' fields

        Returns:
            GenerationResult object
        """
        pass

    def is_initialized(self) -> bool:
        """Check if generator has been initialized."""
        return self._initialized

    def _ensure_initialized(self):
        """Raise error if generator not initialized."""
        if not self._initialized:
            raise RuntimeError(
                f"{self.__class__.__name__} not initialized. " "Call initialization first."
            )
