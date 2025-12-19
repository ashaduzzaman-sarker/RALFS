# ============================================================================
# File: tests/test_generator_factory.py
# ============================================================================
"""Tests for generator factory."""

import pytest
from ralfs.generator.factory import create_generator
from ralfs.generator.fid import FiDGenerator
from ralfs.core.config import RALFSConfig, GeneratorConfig


@pytest.fixture
def base_config():
    """Create base configuration."""
    config = RALFSConfig()
    config.generator = GeneratorConfig(
        model_name="google/flan-t5-small",
    )
    return config


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
