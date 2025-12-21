# ============================================================================
# File: ralfs/generator/factory.py
# ============================================================================
"""Factory for creating generators."""


from omegaconf import DictConfig

from ralfs.core.logging import get_logger
from ralfs.generator.base import BaseGenerator
from ralfs.generator.fid import FiDGenerator

logger = get_logger(__name__)


def create_generator(
    cfg: DictConfig,
    generator_type: str | None = None,
) -> BaseGenerator:
    """
    Factory function to create generator based on configuration.

    Args:
        cfg: Configuration object
        generator_type: Type of generator ('fid')
                       If None, uses cfg.generator.type

    Returns:
        Initialized generator instance

    Example:
        >>> from ralfs.core import load_config
        >>> from ralfs.generator import create_generator
        >>>
        >>> config = load_config("configs/ralfs.yaml")
        >>> generator = create_generator(config)
        >>> result = generator.generate("query", passages)
    """
    # Get generator type from config or argument
    if generator_type is None:
        generator_type = getattr(cfg.generator, "type", "fid")

    generator_type = generator_type.lower()

    logger.info(f"Creating {generator_type} generator...")

    # Create generator
    if generator_type == "fid":
        generator = FiDGenerator(cfg)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}. " f"Choose from: fid")

    logger.info(f"✓ {generator_type.upper()} generator created")
    return generator
