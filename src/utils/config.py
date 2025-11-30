"""RALFS Configuration Utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Union
from omegaconf import OmegaConf, DictConfig


def load_config(path: Union[str, Path]) -> DictConfig:
    """Load any YAML config file manually.
    
    Args:
        path: Path to YAML config file
        
    Returns:
        DictConfig containing loaded configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file is not valid YAML
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        return OmegaConf.load(config_path)
    except Exception as e:
        raise ValueError(f"Invalid YAML config at {config_path}: {str(e)}")


def print_config(cfg: DictConfig, logger=None, resolve: bool = True) -> None:
    """Pretty print configuration for debugging."""
    if logger:
        logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg, resolve=resolve))
    else:
        print(OmegaConf.to_yaml(cfg, resolve=resolve))