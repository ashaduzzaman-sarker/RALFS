# ============================================================================
# File: tests/test_fid_generator.py
# ============================================================================
"""Tests for FiD generator."""

import pytest
from ralfs.generator.fid import FiDGenerator
from ralfs.generator.base import GenerationResult
from ralfs.core.config import RALFSConfig, GeneratorConfig


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = RALFSConfig()
    config.generator = GeneratorConfig(
        model_name="google/flan-t5-small",  # Small model for testing
        max_input_length=256,
        max_output_length=50,
        num_beams=2,
    )
    return config


@pytest.fixture
def sample_passages():
    """Create sample passages for testing."""
    return [
        {"text": "Machine learning is a subset of AI.", "score": 0.95},
        {"text": "Deep learning uses neural networks.", "score": 0.90},
        {"text": "NLP processes human language.", "score": 0.85},
        {"text": "Computer vision analyzes images.", "score": 0.80},
        {"text": "Robotics involves autonomous systems.", "score": 0.70},
    ]


class TestFiDGenerator:
    """Tests for FiDGenerator."""
    
    @pytest.mark.slow
    def test_generator_initialization(self, test_config):
        """Test generator initialization."""
        generator = FiDGenerator(test_config)
        assert generator.is_initialized()
        assert generator.model is not None
        assert generator.tokenizer is not None
    
    @pytest.mark.slow
    def test_generate_basic(self, test_config, sample_passages):
        """Test basic generation."""
        generator = FiDGenerator(test_config)
        result = generator.generate("What is machine learning?", sample_passages)
        
        assert isinstance(result, GenerationResult)
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0
        assert result.k_used > 0
        assert result.k_used <= len(sample_passages)
        assert result.num_passages == len(sample_passages)
    
    @pytest.mark.slow
    def test_generate_with_empty_passages(self, test_config):
        """Test generation with no passages."""
        generator = FiDGenerator(test_config)
        result = generator.generate("test query", [])
        
        assert result.k_used == 0
        assert result.num_passages == 0
        assert "No relevant passages" in result.summary
    
    @pytest.mark.slow
    def test_adaptive_k_selection(self, test_config, sample_passages):
        """Test that adaptive k is working."""
        generator = FiDGenerator(test_config)
        result = generator.generate("test query", sample_passages)
        
        # k should be selected adaptively
        assert result.k_used >= generator.adaptive_k_selector.min_k
        assert result.k_used <= generator.adaptive_k_selector.max_k
        assert result.adaptive_strategy == "score_dropoff"
    
    @pytest.mark.slow
    def test_generate_batch(self, test_config, sample_passages):
        """Test batch generation."""
        generator = FiDGenerator(test_config)
        
        queries = ["What is ML?", "What is NLP?"]
        passages_list = [sample_passages, sample_passages]
        
        results = generator.generate_batch(queries, passages_list)
        
        assert len(results) == 2
        assert all(isinstance(r, GenerationResult) for r in results)
        assert all(len(r.summary) > 0 for r in results)
    
    @pytest.mark.slow
    def test_generate_with_high_k(self, test_config):
        """Test generation with many passages."""
        generator = FiDGenerator(test_config)
        
        # Create many passages
        passages = [
            {"text": f"This is passage number {i}.", "score": 1.0 - i*0.01}
            for i in range(50)
        ]
        
        result = generator.generate("test query", passages)
        
        # Should limit k to max_k
        assert result.k_used <= generator.adaptive_k_selector.max_k

