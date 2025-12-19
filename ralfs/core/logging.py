# ============================================================================
# File: ralfs/core/logging.py
# ============================================================================
"""Logging configuration for RALFS."""

import logging
import sys
from pathlib import Path
from typing import Optional

# Define a consistent formatter for all loggers
RALFS_FORMATTER = logging.Formatter(
    "[%(asctime)s] %(levelname)s %(name)s %(message)s", datefmt="%m/%d/%y %H:%M:%S"
)

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger instance with specified name and level.
    Ensures propagation to the root logger for central handling and testing.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Ensure propagation is True so messages reach the root logger (critical for pytest-caplog)
    logger.propagate = True

    # Add a default StreamHandler if no handlers are present on this specific logger
    # This allows standalone logging even if setup_logging hasn't been called,
    # but still respects propagation.
    if not logger.handlers:
        # By default, StreamHandler logs to sys.stderr, which is usually intercepted by caplog.
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(RALFS_FORMATTER)
        logger.addHandler(handler)
        
    return logger

def setup_logging(log_level: str = "INFO", log_file: Optional[Path | str] = None) -> None:
    """
    Set up global logging configuration.
    Configures handlers for the root logger and suppresses verbose external libraries.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())

    # Clear existing handlers to prevent duplicate output, especially important in Colab/Jupyter
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add a console handler to the root logger
    console_handler = logging.StreamHandler(sys.stdout)  # Direct to stdout for general console output
    console_handler.setFormatter(RALFS_FORMATTER)
    root_logger.addHandler(console_handler)

    if log_file:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(RALFS_FORMATTER)
        root_logger.addHandler(file_handler)

    # Suppress verbose loggers from external libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

