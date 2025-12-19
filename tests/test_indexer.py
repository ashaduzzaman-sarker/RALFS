# ============================================================================
# File: tests/test_indexer.py
# ============================================================================
"""Tests for index builder."""

import pytest
from pathlib import Path
import numpy as np
from ralfs.data.indexer import IndexBuilder, build_index
from ralfs.core.config import RALFSConfig, DataConfig, RetrieverConfig


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = RALFSConfig()
    config.data = DataConfig(
        dataset="arxiv",
        split="train",
        max_samples=5,
    )
    config.retriever = RetrieverConfig(
        dense_model="sentence-transformers/all-MiniLM-L12-v2",
    )
    return config


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
        # First preprocess to create chunks
        from ralfs.data.processor import run_preprocessing
        run_preprocessing(test_config, force_rechunk=True)
        
        # Build index
        builder = IndexBuilder(test_config)
        index_path = builder.build_dense_index(force_rebuild=True)
        
        assert index_path.exists()
        assert (builder.index_dir / "embeddings.npy").exists()
        assert (builder.index_dir / "metadata.json").exists()
    
    @pytest.mark.slow
    def test_build_sparse_index(self, test_config):
        """Test building BM25 index."""
        # First preprocess to create chunks
        from ralfs.data.processor import run_preprocessing
        run_preprocessing(test_config, force_rechunk=True)
        
        # Build index
        builder = IndexBuilder(test_config)
        index_path = builder.build_sparse_index(force_rebuild=True)
        
        assert index_path.exists()
    
    @pytest.mark.slow
    def test_build_all_indexes(self, test_config):
        """Test building all indexes."""
        # First preprocess
        from ralfs.data.processor import run_preprocessing
        run_preprocessing(test_config, force_rechunk=True)
        
        # Build all indexes
        builder = IndexBuilder(test_config)
        indexes = builder.build_all_indexes(force_rebuild=True)
        
        assert 'dense' in indexes
        assert 'sparse' in indexes
        assert all(p.exists() for p in indexes.values())
