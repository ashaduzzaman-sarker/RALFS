# ============================================================================
# File: ralfs/data/chunker.py
# ============================================================================
"""Text chunking strategies for long documents."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Protocol

import nltk

from ralfs.core.logging import get_logger

logger = get_logger(__name__)

# Download NLTK data (only once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


@dataclass
class Chunk:
    """Text chunk data structure."""
    text: str
    chunk_id: str
    doc_id: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)
    
    def __len__(self) -> int:
        """Return chunk text length."""
        return len(self.text)
    
    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Chunk(id={self.chunk_id}, len={len(self.text)}, text='{preview}')"


class ChunkerProtocol(Protocol):
    """Protocol for chunker classes."""
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        ...


class BaseChunker(ABC):
    """Base class for all chunkers."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 128, min_chunk_size: int = None):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            overlap: Overlap size in tokens
            min_chunk_size: Minimum chunk size (discard smaller chunks). Defaults to chunk_size // 5.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Default min_chunk_size to 20% of chunk_size if not specified
        self.min_chunk_size = min_chunk_size if min_chunk_size is not None else max(1, chunk_size // 5)
        
        if overlap >= chunk_size:
            raise ValueError(f"Overlap ({overlap}) must be < chunk_size ({chunk_size})")
        if self.min_chunk_size > chunk_size:
            raise ValueError(f"min_chunk_size ({self.min_chunk_size}) must be <= chunk_size ({chunk_size})")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'"]+', '', text)
        return text
    
    def _count_tokens(self, text: str) -> int:
        """Simple token counting (split by whitespace)."""
        return len(text.split())
    
    @abstractmethod
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Chunk text into smaller pieces.
        
        Args:
            text: Input text to chunk
            doc_id: Document ID for tracking
        
        Returns:
            List of Chunk objects
        """
        pass


class FixedChunker(BaseChunker):
    """Simple fixed-size chunking with sliding window."""
    
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        """Chunk text using fixed-size sliding window."""
        text = self._clean_text(text)
        words = text.split()
        
        if not words:
            logger.warning(f"Empty text for {doc_id}")
            return []
        
        chunks: List[Chunk] = []
        start_idx = 0
        start_char = 0
        
        while start_idx < len(words):
            # Get chunk words
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            
            # Skip if too small
            if len(chunk_words) < self.min_chunk_size and start_idx > 0:
                break
            
            chunk_text = " ".join(chunk_words)
            end_char = start_char + len(chunk_text)
            
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"{doc_id}_c{len(chunks)}",
                doc_id=doc_id,
                start_char=start_char,
                end_char=end_char,
                metadata={
                    "strategy": "fixed",
                    "token_count": len(chunk_words),
                },
            ))
            
            # Move window with overlap
            start_idx += self.chunk_size - self.overlap
            start_char = end_char - len(" ".join(words[end_idx - self.overlap:end_idx]))
        
        logger.debug(f"Fixed chunking: {doc_id} → {len(chunks)} chunks")
        return chunks


class SentenceChunker(BaseChunker):
    """Sentence-boundary aware chunking."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent_tokenizer = nltk.sent_tokenize
    
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        """Chunk text respecting sentence boundaries."""
        text = self._clean_text(text)
        
        try:
            sentences = self.sent_tokenizer(text)
        except Exception as e:
            logger.warning(f"Sentence tokenization failed for {doc_id}: {e}")
            # Fallback to fixed chunking
            return FixedChunker(self.chunk_size, self.overlap, self.min_chunk_size).chunk(text, doc_id)
        
        if not sentences:
            logger.warning(f"No sentences found in {doc_id}")
            return []
        
        chunks: List[Chunk] = []
        current_sents: List[str] = []
        current_tokens = 0
        start_char = 0
        
        for sent in sentences:
            sent_tokens = self._count_tokens(sent)
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_tokens + sent_tokens > self.chunk_size and current_sents:
                chunk_text = " ".join(current_sents)
                end_char = start_char + len(chunk_text)
                
                # Only save if meets minimum size
                if current_tokens >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=chunk_text,
                        chunk_id=f"{doc_id}_c{len(chunks)}",
                        doc_id=doc_id,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={
                            "strategy": "sentence",
                            "token_count": current_tokens,
                            "sentence_count": len(current_sents),
                        },
                    ))
                
                # Keep last few sentences for overlap
                overlap_sents = current_sents[-2:] if len(current_sents) >= 2 else []
                overlap_text = " ".join(overlap_sents)
                overlap_tokens = self._count_tokens(overlap_text)
                
                current_sents = overlap_sents
                current_tokens = overlap_tokens
                start_char = end_char - len(overlap_text)
            
            current_sents.append(sent)
            current_tokens += sent_tokens
        
        # Save remaining chunk
        if current_sents and current_tokens >= self.min_chunk_size:
            chunk_text = " ".join(current_sents)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"{doc_id}_c{len(chunks)}",
                doc_id=doc_id,
                start_char=start_char,
                end_char=len(text),
                metadata={
                    "strategy": "sentence",
                    "token_count": current_tokens,
                    "sentence_count": len(current_sents),
                },
            ))
        
        logger.debug(f"Sentence chunking: {doc_id} → {len(chunks)} chunks")
        return chunks


class SemanticChunker(SentenceChunker):
    """
    Semantic chunking with sentence boundaries (alias for SentenceChunker).
    
    Note: For true semantic chunking with embeddings, extend this class
    to compute sentence embeddings and group similar sentences.
    """
    
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Chunk text using sentence boundaries (semantic-aware).
        
        Future enhancement: Use sentence embeddings to group semantically
        similar sentences together before chunking.
        """
        chunks = super().chunk(text, doc_id)
        
        # Update metadata
        for chunk in chunks:
            chunk.metadata["strategy"] = "semantic"
        
        logger.debug(f"Semantic chunking: {doc_id} → {len(chunks)} chunks")
        return chunks


def create_chunker(
    strategy: str = "semantic",
    chunk_size: int = 512,
    overlap: int = 128,
    min_chunk_size: int = 100,
) -> BaseChunker:
    """
    Factory function to create chunker based on strategy.
    
    Args:
        strategy: Chunking strategy ('fixed', 'sentence', 'semantic')
        chunk_size: Target chunk size in tokens
        overlap: Overlap size in tokens
        min_chunk_size: Minimum chunk size
    
    Returns:
        Chunker instance
    
    Raises:
        ValueError: If strategy is invalid
    """
    chunkers = {
        "fixed": FixedChunker,
        "sentence": SentenceChunker,
        "semantic": SemanticChunker,
    }
    
    if strategy not in chunkers:
        raise ValueError(
            f"Invalid chunking strategy: {strategy}. "
            f"Choose from: {list(chunkers.keys())}"
        )
    
    return chunkers[strategy](
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_size=min_chunk_size,
    )
