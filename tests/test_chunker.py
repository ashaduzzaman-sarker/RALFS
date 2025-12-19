"""Tests for text chunkers."""

import pytest
from ralfs.data.chunker import (
    Chunk,
    FixedChunker,
    SentenceChunker,
    SemanticChunker,
    create_chunker,
)


class TestChunk:
    """Tests for Chunk dataclass."""
    
    def test_chunk_creation(self):
        """Test creating a chunk."""
        chunk = Chunk(
            text="This is a test chunk.",
            chunk_id="doc1_c0",
            doc_id="doc1",
            start_char=0,
            end_char=21,
        )
        assert chunk.chunk_id == "doc1_c0"
        assert len(chunk) == 21
    
    def test_chunk_to_dict(self):
        """Test converting chunk to dict."""
        chunk = Chunk(
            text="Test",
            chunk_id="c1",
            doc_id="d1",
            start_char=0,
            end_char=4,
            metadata={"key": "value"},
        )
        chunk_dict = chunk.to_dict()
        assert isinstance(chunk_dict, dict)
        assert chunk_dict["metadata"]["key"] == "value"


class TestFixedChunker:
    """Tests for FixedChunker."""
    
    def test_fixed_chunking(self):
        """Test basic fixed chunking."""
        text = " ".join(["word"] * 100)  # 100 words
        chunker = FixedChunker(chunk_size=30, overlap=10, min_chunk_size=10)
        
        chunks = chunker.chunk(text, "doc1")
        
        assert len(chunks) > 0
        assert all(c.doc_id == "doc1" for c in chunks)
        assert all("fixed" in c.metadata["strategy"] for c in chunks)
    
    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = FixedChunker()
        chunks = chunker.chunk("", "doc1")
        assert len(chunks) == 0
    
    def test_chunk_ids_unique(self):
        """Test that chunk IDs are unique."""
        text = " ".join(["word"] * 100)
        chunker = FixedChunker(chunk_size=20, overlap=5, min_chunk_size=10)
        chunks = chunker.chunk(text, "doc1")
        
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))


class TestSentenceChunker:
    """Tests for SentenceChunker."""
    
    def test_sentence_chunking(self):
        """Test sentence-based chunking."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunker = SentenceChunker(chunk_size=10, overlap=3, min_chunk_size=5)
        
        chunks = chunker.chunk(text, "doc1")
        
        assert len(chunks) > 0
        assert all(c.doc_id == "doc1" for c in chunks)
        assert all("sentence" in c.metadata["strategy"] for c in chunks)
    
    def test_respects_sentence_boundaries(self):
        """Test that chunks respect sentence boundaries."""
        text = "First. Second. Third. Fourth. Fifth."
        chunker = SentenceChunker(chunk_size=15, overlap=5, min_chunk_size=5)
        chunks = chunker.chunk(text, "doc1")
        
        # Each chunk should contain complete sentences
        for chunk in chunks:
            # Should not have incomplete sentences (ending with partial word)
            assert not chunk.text.endswith(" ")


class TestSemanticChunker:
    """Tests for SemanticChunker."""
    
    def test_semantic_chunking(self):
        """Test semantic chunking (currently uses sentence boundaries)."""
        text = "Sentence one here. Sentence two here. Sentence three here."
        chunker = SemanticChunker(chunk_size=15, overlap=5, min_chunk_size=5)
        
        chunks = chunker.chunk(text, "doc1")
        
        assert len(chunks) > 0
        assert all("semantic" in c.metadata["strategy"] for c in chunks)


class TestCreateChunker:
    """Tests for chunker factory."""
    
    def test_create_fixed_chunker(self):
        """Test creating fixed chunker."""
        chunker = create_chunker(strategy="fixed")
        assert isinstance(chunker, FixedChunker)
    
    def test_create_sentence_chunker(self):
        """Test creating sentence chunker."""
        chunker = create_chunker(strategy="sentence")
        assert isinstance(chunker, SentenceChunker)
    
    def test_create_semantic_chunker(self):
        """Test creating semantic chunker."""
        chunker = create_chunker(strategy="semantic")
        assert isinstance(chunker, SemanticChunker)
    
    def test_invalid_strategy_raises_error(self):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid chunking strategy"):
            create_chunker(strategy="invalid")
    
    def test_chunker_with_parameters(self):
        """Test creating chunker with custom parameters."""
        chunker = create_chunker(
            strategy="fixed",
            chunk_size=256,
            overlap=64,
            min_chunk_size=50,
        )
        assert chunker.chunk_size == 256
        assert chunker.overlap == 64
        assert chunker.min_chunk_size == 50