# ============================================================================
# File: tests/test_retriever.py
# ============================================================================
"""Consolidated tests for retriever components: base, dense, sparse, hybrid, reranker."""

import pytest
from ralfs.retriever.base import RetrievalResult, BaseRetriever
from ralfs.retriever.dense import DenseRetriever
from ralfs.retriever.sparse import SparseRetriever
from ralfs.retriever.hybrid import HybridRetriever
from ralfs.retriever.reranker import CrossEncoderReranker
from ralfs.core.config import RALFSConfig, DataConfig, RetrieverConfig


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def base_config():
    """Create basic test configuration."""
    config = RALFSConfig()
    config.data = DataConfig(dataset="arxiv", split="train", max_samples=5)
    config.retriever = RetrieverConfig()
    return config


@pytest.fixture
def dense_config():
    """Create configuration for dense retriever."""
    config = RALFSConfig()
    config.data = DataConfig(dataset="arxiv", split="train", max_samples=20)
    config.retriever = RetrieverConfig(
        dense_model="sentence-transformers/all-MiniLM-L12-v2",
        k_dense=10,
    )
    return config


@pytest.fixture
def hybrid_config():
    """Create configuration for hybrid retriever."""
    config = RALFSConfig()
    config.data = DataConfig(dataset="arxiv", split="train", max_samples=20)
    config.retriever = RetrieverConfig(
        type="hybrid",
        dense_model="sentence-transformers/all-MiniLM-L12-v2",
        k_dense=20,
        k_sparse=20,
        k_final=10,
    )
    return config


@pytest.fixture
def reranker_config():
    """Create configuration for reranker."""
    config = RALFSConfig()
    config.retriever = RetrieverConfig(
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    return config


# ============================================================================
# RetrievalResult Tests
# ============================================================================

class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a retrieval result."""
        result = RetrievalResult(
            text="Test text",
            score=0.95,
            rank=1,
            doc_id="doc_1",
            chunk_id="doc_1_c0",
        )
        assert result.text == "Test text"
        assert result.score == 0.95
        assert result.rank == 1
    
    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = RetrievalResult(
            text="Test",
            score=0.9,
            rank=1,
            metadata={"key": "value"},
        )
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["metadata"]["key"] == "value"
    
    def test_result_repr(self):
        """Test result string representation."""
        result = RetrievalResult(
            text="A" * 100,
            score=0.85,
            rank=2,
        )
        repr_str = repr(result)
        assert "rank=2" in repr_str
        assert "score=0.85" in repr_str


# ============================================================================
# DenseRetriever Tests
# ============================================================================

class TestDenseRetriever:
    """Tests for DenseRetriever."""
    
    def test_retriever_initialization(self, dense_config):
        """Test retriever initialization."""
        retriever = DenseRetriever(dense_config)
        assert retriever.model is not None
        assert retriever.k_default == 10
    
    @pytest.mark.slow
    def test_load_index(self, dense_config):
        """Test loading FAISS index."""
        from ralfs.data.processor import run_preprocessing
        from ralfs.data.indexer import build_index
        
        run_preprocessing(dense_config, force_rechunk=True)
        build_index(dense_config, force_rebuild=True)
        
        retriever = DenseRetriever(dense_config)
        retriever.load_index()
        
        assert retriever.is_initialized()
        assert retriever.index is not None
    
    @pytest.mark.slow
    def test_retrieve(self, dense_config):
        """Test dense retrieval."""
        from ralfs.data.processor import run_preprocessing
        from ralfs.data.indexer import build_index
        
        run_preprocessing(dense_config, force_rechunk=True)
        build_index(dense_config, force_rebuild=True)
        
        retriever = DenseRetriever(dense_config)
        retriever.load_index()
        results = retriever.retrieve("semiparametric regression", k=5)
        
        assert len(results) <= 5
        assert all(r.score > 0 for r in results)
        assert results[0].rank == 1


# ============================================================================
# SparseRetriever Tests
# ============================================================================

class TestSparseRetriever:
    """Tests for SparseRetriever."""
    
    def test_retriever_initialization(self, base_config):
        """Test retriever initialization."""
        base_config.retriever.k_sparse = 10
        retriever = SparseRetriever(base_config)
        assert retriever.k_default == 10
        assert retriever.k1 == 1.5
        assert retriever.b == 0.75
    
    def test_build_index_from_texts(self, base_config):
        """Test building BM25 index from texts."""
        texts = ["This is document one.", "This is document two."]
        retriever = SparseRetriever(base_config, chunks=texts)
        
        assert retriever.is_initialized()
        assert retriever.bm25 is not None
    
    @pytest.mark.slow
    def test_load_index(self, base_config):
        """Test loading chunks and building BM25."""
        from ralfs.data.processor import run_preprocessing
        
        run_preprocessing(base_config, force_rechunk=True)
        
        retriever = SparseRetriever(base_config)
        retriever.load_index()
        
        assert retriever.is_initialized()
        assert retriever.bm25 is not None
    
    @pytest.mark.slow
    def test_retrieve(self, base_config):
        """Test sparse retrieval."""
        from ralfs.data.processor import run_preprocessing
        
        run_preprocessing(base_config, force_rechunk=True)
        
        retriever = SparseRetriever(base_config)
        retriever.load_index()
        results = retriever.retrieve("quantum physics", k=5)
        
        assert len(results) <= 5
        assert all(r.score >= 0 for r in results)


# ============================================================================
# HybridRetriever Tests
# ============================================================================

class TestHybridRetriever:
    """Tests for HybridRetriever."""
    
    def test_retriever_initialization(self, hybrid_config):
        """Test hybrid retriever initialization."""
        retriever = HybridRetriever(hybrid_config)
        
        assert retriever.dense is not None
        assert retriever.sparse is not None
        assert retriever.reranker is not None
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_load_index(self, hybrid_config):
        """Test loading all indexes."""
        from ralfs.data.processor import run_preprocessing
        from ralfs.data.indexer import build_index
        
        run_preprocessing(hybrid_config, force_rechunk=True)
        build_index(hybrid_config, force_rebuild=True)
        
        retriever = HybridRetriever(hybrid_config)
        retriever.load_index()
        
        assert retriever.is_initialized()
        assert retriever.dense.is_initialized()
        assert retriever.sparse.is_initialized()
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_hybrid_retrieve(self, hybrid_config):
        """Test full hybrid retrieval pipeline."""
        from ralfs.data.processor import run_preprocessing
        from ralfs.data.indexer import build_index
        
        run_preprocessing(hybrid_config, force_rechunk=True)
        build_index(hybrid_config, force_rebuild=True)
        
        retriever = HybridRetriever(hybrid_config)
        retriever.load_index()
        results = retriever.retrieve("semiparametric regression", k=5)
        
        assert len(results) <= 5
        assert all(r.score >= 0 for r in results)
        assert results[0].rank == 1
        
        if results:
            assert results[0].metadata.get("reranked") is True


# ============================================================================
# CrossEncoderReranker Tests
# ============================================================================

class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""
    
    def test_reranker_initialization(self, reranker_config):
        """Test reranker initialization."""
        reranker = CrossEncoderReranker(reranker_config)
        assert reranker.model is not None
        assert reranker.enabled is True
    
    def test_reranker_disabled(self):
        """Test reranker when disabled."""
        config = RALFSConfig()
        config.retriever = RetrieverConfig()
        config.retriever.reranker = {"enabled": False}
        
        reranker = CrossEncoderReranker(config)
        assert reranker.enabled is False
    
    def test_rerank_results(self, reranker_config):
        """Test reranking results."""
        reranker = CrossEncoderReranker(reranker_config)
        
        candidates = [
            RetrievalResult(text="The cat sat on the mat", score=0.5, rank=1),
            RetrievalResult(text="Dogs are great pets", score=0.6, rank=2),
            RetrievalResult(text="Cats and dogs live together", score=0.4, rank=3),
        ]
        
        query = "cats and pets"
        reranked = reranker.rerank(query, candidates, top_k=2)
        
        assert len(reranked) == 2
        assert reranked[0].rank == 1
        assert reranked[0].metadata["reranked"] is True
    
    def test_rerank_dicts(self, reranker_config):
        """Test reranking dict candidates."""
        reranker = CrossEncoderReranker(reranker_config)
        
        candidates = [
            {"text": "Machine learning is great", "score": 0.5},
            {"text": "Deep learning uses neural networks", "score": 0.6},
        ]
        
        query = "neural networks"
        reranked = reranker.rerank(query, candidates, top_k=2)
        
        assert len(reranked) == 2
        assert all(isinstance(r, RetrievalResult) for r in reranked)
