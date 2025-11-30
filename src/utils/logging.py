"""RALFS Logging Utilities."""
from __future__ import annotations

import logging
from typing import Optional
from omegaconf import DictConfig


def get_logger(name: str = "RALFS", cfg: Optional[DictConfig] = None) -> logging.Logger:
    """Returns a configured logger.
    
    Safe against hydra multiple reinitializations.
    
    Args:
        name: Logger name (default: "RALFS")
        cfg: Optional Hydra config for log level
        
    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers when Hydra re-runs code
    if logger.handlers:
        return logger
    
    # Set log level from config or default to INFO
    log_level = logging.INFO
    if cfg and "logging" in cfg and "level" in cfg.logging:
        level_str = str(cfg.logging.level).upper()
        log_level = getattr(logging, level_str, logging.INFO)
    
    logger.setLevel(log_level)
    
    # Configure handler and formatter
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    
    return logger