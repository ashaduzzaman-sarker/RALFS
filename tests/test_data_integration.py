# ============================================================================
# File: tests/test_data_integration.py
# ============================================================================
"""Integration tests for full data pipeline."""

import pytest

from ralfs.core.config import DataConfig, RALFSConfig, RetrieverConfig
from ralfs.data.indexer import build_index
from ralfs.data.processor import run_preprocessing


@pytest.fixture
def integration_config():
    """Create configuration for integration test."""
    config = RALFSConfig()
    config.data = DataConfig(
        dataset="arxiv",
        split="train",
        max_samples=5,  # Very small for fast testing
        chunk_size=128,
        overlap=32,
        chunking_strategy="sentence",
    )
    config.retriever = RetrieverConfig(
        dense_model="sentence-transformers/all-MiniLM-L12-v2",
    )
    return config


@pytest.mark.slow
@pytest.mark.integration
class TestDataPipelineIntegration:
    """Integration tests for complete data pipeline."""

    def test_full_pipeline(self, integration_config):
        """Test complete pipeline: download → chunk → index."""
        # Step 1: Preprocess
        chunks_path = run_preprocessing(
            integration_config,
            force_download=True,
            force_rechunk=True,
        )
        assert chunks_path.exists()

        # Step 2: Build indexes
        indexes = build_index(integration_config, force_rebuild=True)
        assert "dense" in indexes
        assert "sparse" in indexes

        # Verify all files exist
        assert indexes["dense"].exists()
        assert indexes["sparse"].exists()

    def test_caching_works(self, integration_config):
        """Test that caching prevents re-download."""
        # First run
        chunks_path1 = run_preprocessing(
            integration_config,
            force_download=True,
            force_rechunk=True,
        )

        # Second run (should use cache)
        chunks_path2 = run_preprocessing(
            integration_config,
            force_download=False,
            force_rechunk=False,
        )

        assert chunks_path1 == chunks_path2
