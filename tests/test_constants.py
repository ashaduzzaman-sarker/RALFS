"""Tests for ralfs.core.constants module."""

from pathlib import Path

from ralfs.core.constants import (
    DATA_DIR,
    DEFAULT_DENSE_MODEL,
    DEFAULT_K_FINAL,
    INDEX_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    ROOT_DIR,
    SUPPORTED_DATASETS,
)


class TestPaths:
    """Tests for path constants."""

    def test_root_dir_exists(self):
        """Test that root directory is valid."""
        assert isinstance(ROOT_DIR, Path)
        # Note: May not exist in test environment, just check it's a Path

    def test_data_directories_are_paths(self):
        """Test that data directories are Path objects."""
        assert isinstance(DATA_DIR, Path)
        assert isinstance(RAW_DIR, Path)
        assert isinstance(PROCESSED_DIR, Path)
        assert isinstance(INDEX_DIR, Path)

    def test_data_directories_created(self):
        """Test that data directories are created on import."""
        # These should be created when constants module is imported
        assert DATA_DIR.exists()
        assert RAW_DIR.exists()
        assert PROCESSED_DIR.exists()


class TestModelDefaults:
    """Tests for model default constants."""

    def test_model_names_are_strings(self):
        """Test that model names are strings."""
        assert isinstance(DEFAULT_DENSE_MODEL, str)
        assert len(DEFAULT_DENSE_MODEL) > 0

    def test_k_values_are_integers(self):
        """Test that k values are integers."""
        assert isinstance(DEFAULT_K_FINAL, int)
        assert DEFAULT_K_FINAL > 0


class TestDatasetConfigs:
    """Tests for dataset configurations."""

    def test_supported_datasets_list(self):
        """Test that SUPPORTED_DATASETS is a list."""
        assert isinstance(SUPPORTED_DATASETS, list)
        assert len(SUPPORTED_DATASETS) > 0

    def test_supported_datasets_contains_arxiv(self):
        """Test that arXiv is in supported datasets."""
        assert "arxiv" in SUPPORTED_DATASETS
