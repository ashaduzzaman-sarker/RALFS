"""tests/test_chunker.py
Tests for Semantic Chunking."""
import pytest
from src.data.preprocess import SemanticChunker, Chunk
from src.utils.logging import get_logger


@pytest.fixture
def sample_text():
    return """
    This is the first sentence. This is the second sentence. 
    This is a new paragraph with the third sentence.
    
    Another paragraph starts here. This is sentence four and five are related.
    Sentence six continues the thought.
    """


def test_chunker_basic(sample_text):
    """Test basic chunking functionality."""
    chunker = SemanticChunker(chunk_size=20, overlap=5)
    chunks = chunker.chunk_document(sample_text, "test_doc")
    
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.tokens <= 20 for c in chunks)
    assert chunks[0].chunk_id.startswith("test_doc")


def test_chunker_overlap(sample_text):
    """Test overlap functionality."""
    chunker = SemanticChunker(chunk_size=15, overlap=10)
    chunks = chunker.chunk_document(sample_text, "test_doc")
    
    if len(chunks) > 1:
        # Check that chunks overlap
        first_chunk_end = chunks[0].end_pos
        second_chunk_start = chunks[1].start_pos
        assert first_chunk_end - second_chunk_start >= 10


def test_clean_text():
    """Test text cleaning."""
    chunker = SemanticChunker()
    dirty_text = "  Multiple   spaces\n\nand\n\n\nlines  "
    cleaned = chunker._clean_text(dirty_text)
    assert "  " not in cleaned
    assert cleaned.strip() == "Multiple spaces and lines"
