# ============================================================================
# File: tests/test_downloader.py
# ============================================================================
"""Tests for data downloader."""

import pytest
from pathlib import Path
from ralfs.data.downloader import DatasetDownloader, Document


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
        # Create test documents
        docs = [
            Document(
                id=f"test_{i}",
                text=f"Text {i}",
                summary=f"Summary {i}",
            )
            for i in range(3)
        ]
        
        # Save
        output_path = DatasetDownloader.save(
            docs,
            dataset_name="test",
            split="train",
            output_dir=tmp_path,
        )
        
        assert output_path.exists()
        
        # Load
        loaded_docs = DatasetDownloader.load(
            dataset_name="test",
            split="train",
            input_dir=tmp_path,
        )
        
        assert len(loaded_docs) == 3
        assert loaded_docs[0].id == "test_0"
