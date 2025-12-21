# ============================================================================
# File: tests/test_generator.py
# ============================================================================
"""Consolidated tests for generator components: base, factory, fid, adaptive_k."""

import pytest

from ralfs.core.config import GeneratorConfig, RALFSConfig
from ralfs.generator.adaptive_k import AdaptiveKSelector
from ralfs.generator.base import GenerationResult
from ralfs.generator.factory import create_generator
from ralfs.generator.fid import FiDGenerator

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def base_config():
    """Create base configuration."""
    config = RALFSConfig()
    config.generator = GeneratorConfig(
        model_name="google/flan-t5-small",
    )
    return config


@pytest.fixture
def test_config():
    """Create test configuration for FiD."""
    config = RALFSConfig()
    config.generator = GeneratorConfig(
        model_name="google/flan-t5-small",
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


# ============================================================================
# GenerationResult Tests
# ============================================================================


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_result_creation(self):
        """Test creating a generation result."""
        result = GenerationResult(
            summary="This is a test summary.",
            query="test query",
            k_used=10,
            num_passages=20,
        )
        assert result.summary == "This is a test summary."
        assert result.k_used == 10
        assert result.num_passages == 20

    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = GenerationResult(
            summary="Test",
            query="query",
            k_used=5,
            num_passages=10,
            metadata={"key": "value"},
        )
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["metadata"]["key"] == "value"

    def test_result_repr(self):
        """Test result string representation."""
        result = GenerationResult(
            summary="A" * 200,
            query="test",
            k_used=10,
            num_passages=20,
        )
        repr_str = repr(result)
        assert "k=10/20" in repr_str


# ============================================================================
# Generator Factory Tests
# ============================================================================


class TestGeneratorFactory:
    """Tests for generator factory."""

    @pytest.mark.slow
    def test_create_fid_generator(self, base_config):
        """Test creating FiD generator."""
        generator = create_generator(base_config, generator_type="fid")
        assert isinstance(generator, FiDGenerator)

    @pytest.mark.slow
    def test_create_from_config_type(self, base_config):
        """Test creating generator from config type."""
        base_config.generator.type = "fid"
        generator = create_generator(base_config)
        assert isinstance(generator, FiDGenerator)

    def test_invalid_type(self, base_config):
        """Test creating generator with invalid type."""
        with pytest.raises(ValueError, match="Unknown generator type"):
            create_generator(base_config, generator_type="invalid")


# ============================================================================
# FiDGenerator Tests
# ============================================================================


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
        """Test generation with empty passages."""
        generator = FiDGenerator(test_config)
        result = generator.generate("Test query", [])

        assert isinstance(result, GenerationResult)
        assert result.k_used == 0
        assert result.num_passages == 0

    @pytest.mark.slow
    def test_generate_with_k_limit(self, test_config, sample_passages):
        """Test generation with k limit."""
        test_config.generator.max_k = 3
        generator = FiDGenerator(test_config)
        result = generator.generate("Test query", sample_passages)

        assert result.k_used <= 3


# ============================================================================
# AdaptiveKSelector Tests
# ============================================================================


class TestAdaptiveKSelector:
    """Tests for AdaptiveKSelector."""

    def test_selector_initialization(self):
        """Test selector initialization."""
        selector = AdaptiveKSelector(min_k=5, max_k=30, default_k=20)
        assert selector.min_k == 5
        assert selector.max_k == 30
        assert selector.default_k == 20

    def test_invalid_k_range(self):
        """Test that invalid k range raises error."""
        with pytest.raises(ValueError):
            AdaptiveKSelector(min_k=30, max_k=10)

    def test_select_k_with_empty_scores(self):
        """Test k selection with empty scores."""
        selector = AdaptiveKSelector(default_k=20)
        k = selector.select_k([])
        assert k == 20

    def test_select_k_with_few_passages(self):
        """Test k selection when fewer passages than min_k."""
        selector = AdaptiveKSelector(min_k=10)
        scores = [0.9, 0.8, 0.7]
        k = selector.select_k(scores)
        assert k == 3

    def test_score_dropoff_strategy(self):
        """Test score dropoff strategy."""
        selector = AdaptiveKSelector(min_k=3, max_k=10, strategy="dropoff", threshold=0.1)
        scores = [0.9, 0.88, 0.85, 0.5, 0.45, 0.4]
        k = selector.select_k(scores)
        assert k >= 3
        assert k <= len(scores)

    def test_threshold_strategy(self):
        """Test threshold strategy."""
        selector = AdaptiveKSelector(min_k=2, strategy="threshold", threshold=0.7)
        scores = [0.95, 0.85, 0.75, 0.65, 0.55]
        k = selector.select_k(scores)
        assert k == 3

    def test_percentile_strategy(self):
        """Test percentile strategy."""
        selector = AdaptiveKSelector(min_k=3, strategy="percentile", percentile=75)
        scores = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
        k = selector.select_k(scores)
        assert k >= 3
