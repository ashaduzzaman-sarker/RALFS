# ============================================================================
# File: tests/test_data.py
# ============================================================================
"""Consolidated tests for data pipeline: downloader, chunker, processor, indexer."""

import pytest
import numpy as np
from pathlib import Path
from ralfs.data.downloader import DatasetDownloader, Document
from ralfs.data.chunker import (
    Chunk,
    FixedChunker,
    SentenceChunker,
    SemanticChunker,
    create_chunker,
)
from ralfs.data.processor import DocumentProcessor, run_preprocessing
from ralfs.data.indexer import IndexBuilder, build_index
from ralfs.core.config import RALFSConfig, DataConfig, RetrieverConfig


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_config():
    """Create test configuration."""
    config = RALFSConfig()
    config.data = DataConfig(
        dataset="arxiv",
        split="train",
        max_samples=5,
        chunk_size=256,
        overlap=64,
        chunking_strategy="sentence",
    )
    config.retriever = RetrieverConfig(
        dense_model="sentence-transformers/all-MiniLM-L12-v2",
    )
    return config


# ============================================================================
# Document Tests
# ============================================================================

class TestDocument:
    """Tests for Document dataclass."""
    
    def test_document_creation(self):
        """Test creating a document."""
        doc = Document(
            id="test_1",
            text="This is a test document.",
            summary="Test summary",
            title="Test Doc",
            domain="test",
            source="test",
        )
        assert doc.id == "test_1"
        assert len(doc) == len("This is a test document.")
    
    def test_document_to_dict(self):
        """Test converting document to dict."""
        doc = Document(
            id="test_1",
            text="Test text",
            summary="Test summary",
        )
        doc_dict = doc.to_dict()
        assert isinstance(doc_dict, dict)
        assert doc_dict["id"] == "test_1"
        assert doc_dict["text"] == "Test text"


# ============================================================================
# Downloader Tests
# ============================================================================

class TestDatasetDownloader:
    """Tests for DatasetDownloader."""
    
    def test_list_supported_datasets(self):
        """Test listing supported datasets."""
        datasets = DatasetDownloader.list_supported()
        assert isinstance(datasets, list)
        assert "arxiv" in datasets
        assert "govreport" in datasets
        assert "booksum" in datasets
    
    def test_unsupported_dataset_raises_error(self):
        """Test that unsupported dataset raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            DatasetDownloader.download("invalid_dataset")
    
    @pytest.mark.slow
    def test_download_arxiv_sample(self):
        """Test downloading small arxiv sample."""
        docs = DatasetDownloader.download(
            dataset_name="arxiv",
            split="train",
            max_samples=5,
        )
        assert len(docs) == 5
        assert all(isinstance(d, Document) for d in docs)
        assert all(d.domain == "scientific" for d in docs)
    
    def test_save_and_load_documents(self, tmp_path):
        """Test saving and loading documents."""
        docs = [
            Document(
                id=f"test_{i}",
                text=f"Text {i}",
                summary=f"Summary {i}",
            )
            for i in range(3)
        ]
        
        output_path = DatasetDownloader.save(
            docs,
            dataset_name="test",
            split="train",
            output_dir=tmp_path,
        )
        
        assert output_path.exists()
        
        loaded_docs = DatasetDownloader.load(
            dataset_name="test",
            split="train",
            input_dir=tmp_path,
        )
        
        assert len(loaded_docs) == 3
        assert loaded_docs[0].id == "test_0"


# ============================================================================
# Chunk Tests
# ============================================================================

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


# ============================================================================
# Chunker Tests
# ============================================================================

class TestFixedChunker:
    """Tests for FixedChunker."""
    
    def test_fixed_chunking(self):
        """Test basic fixed chunking."""
        text = " ".join(["word"] * 100)
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
        
        for chunk in chunks:
            assert not chunk.text.endswith(" ")


class TestSemanticChunker:
    """Tests for SemanticChunker."""
    
    def test_semantic_chunking(self):
        """Test semantic chunking."""
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


# ============================================================================
# Processor Tests
# ============================================================================

class TestDocumentProcessor:
    """Tests for DocumentProcessor."""
    
    def test_processor_initialization(self, test_config):
        """Test processor initialization."""
        processor = DocumentProcessor(test_config)
        assert processor.dataset_name == "arxiv"
        assert processor.split == "train"
        assert processor.max_samples == 5
    
    @pytest.mark.slow
    def test_download_documents(self, test_config):
        """Test downloading documents."""
        processor = DocumentProcessor(test_config)
        docs = processor.download_documents()
        
        assert len(docs) > 0
        assert len(docs) <= 5
    
    @pytest.mark.slow
    def test_chunk_documents(self, test_config):
        """Test chunking documents."""
        processor = DocumentProcessor(test_config)
        docs = processor.download_documents()
        chunks = processor.chunk_documents(docs)
        
        assert len(chunks) > 0
        assert all(c.metadata["source"] == "arxiv" for c in chunks)
    
    @pytest.mark.slow
    def test_full_pipeline(self, test_config):
        """Test full preprocessing pipeline."""
        processor = DocumentProcessor(test_config)
        output_path = processor.process(force_download=True, force_rechunk=True)
        assert output_path.exists()


class TestRunPreprocessing:
    """Tests for run_preprocessing function."""
    
    @pytest.mark.slow
    def test_run_preprocessing(self, test_config):
        """Test backward-compatible preprocessing function."""
        output_path = run_preprocessing(test_config, force_rechunk=True)
        assert output_path.exists()


# ============================================================================
# Indexer Tests
# ============================================================================

class TestIndexBuilder:
    """Tests for IndexBuilder."""
    
    def test_builder_initialization(self, test_config):
        """Test builder initialization."""
        builder = IndexBuilder(test_config)
        assert builder.dataset_name == "arxiv"
        assert builder.index_dir.exists()
    
    @pytest.mark.slow
    def test_build_dense_index(self, test_config):
        """Test building dense FAISS index."""
        run_preprocessing(test_config, force_rechunk=True)
        
        builder = IndexBuilder(test_config)
        index_path = builder.build_dense_index(force_rebuild=True)
        
        assert index_path.exists()
        assert (builder.index_dir / "embeddings.npy").exists()
        assert (builder.index_dir / "metadata.json").exists()
    
    @pytest.mark.slow
    def test_build_sparse_index(self, test_config):
        """Test building BM25 index."""
        run_preprocessing(test_config, force_rechunk=True)
        
        builder = IndexBuilder(test_config)
        index_path = builder.build_sparse_index(force_rebuild=True)
        
        assert index_path.exists()
    
    @pytest.mark.slow
    def test_build_all_indexes(self, test_config):
        """Test building all indexes."""
        run_preprocessing(test_config, force_rechunk=True)
        
        builder = IndexBuilder(test_config)
        indexes = builder.build_all_indexes(force_rebuild=True)
        
        assert 'dense' in indexes
        assert 'sparse' in indexes
        assert all(p.exists() for p in indexes.values())
