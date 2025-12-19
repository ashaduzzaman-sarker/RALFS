# ============================================================================
# File: tests/test_processor.py
# ============================================================================
"""Tests for document processor."""

import pytest
from pathlib import Path
from ralfs.data.processor import DocumentProcessor, run_preprocessing
from ralfs.core.config import RALFSConfig, DataConfig


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = RALFSConfig()
    config.data = DataConfig(
        dataset="arxiv",
        split="train",
        max_samples=5,  # Small sample for testing
        chunk_size=256,
        overlap=64,
        chunking_strategy="sentence",
    )
    return config


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
        assert len(docs) <= 5  # max_samples
    
    @pytest.mark.slow
    def test_chunk_documents(self, test_config):
        """Test chunking documents."""
        processor = DocumentProcessor(test_config)
        docs = processor.download_documents()
        chunks = processor.chunk_documents(docs)
        
        assert len(chunks) > 0
        assert all(c.metadata["source"] == "arxiv" for c in chunks)
    
    @pytest.mark.slow
    def test_full_pipeline(self, test_config, tmp_path):
        """Test full preprocessing pipeline."""
        # Modify config to use tmp directory
        from ralfs.core.constants import PROCESSED_DIR
        test_config.data.dataset = "arxiv"
        
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

