# ============================================================================
# File: ralfs/retriever/base.py
# ============================================================================
"""Base retriever class and common utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RetrievalResult:
    """
    Standardized retrieval result.

    Attributes:
        text: Retrieved text content
        score: Relevance score (higher = more relevant)
        rank: Rank in results (1-indexed)
        doc_id: Source document ID
        chunk_id: Chunk ID within document
        metadata: Additional metadata
    """

    text: str
    score: float
    rank: int
    doc_id: str | None = None
    chunk_id: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"RetrievalResult(rank={self.rank}, score={self.score:.4f}, text='{preview}')"


class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.

    All retrievers must implement:
    - retrieve(): Search for relevant chunks
    - load_index(): Load the retrieval index
    """

    def __init__(self, cfg):
        """
        Initialize retriever with configuration.

        Args:
            cfg: Configuration object (RALFSConfig)
        """
        self.cfg = cfg
        self._initialized = False

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            k: Number of results to return (None = use default)

        Returns:
            List of RetrievalResult objects, sorted by score (descending)
        """
        pass

    @abstractmethod
    def load_index(self, index_path: Path | None = None) -> None:
        """
        Load retrieval index from disk.

        Args:
            index_path: Path to index (None = use default from config)
        """
        pass

    def is_initialized(self) -> bool:
        """Check if retriever has been initialized."""
        return self._initialized

    def _ensure_initialized(self):
        """Raise error if retriever not initialized."""
        if not self._initialized:
            raise RuntimeError(
                f"{self.__class__.__name__} not initialized. " "Call load_index() first."
            )
