# src/ralfs/__init__.py
from .core import get_logger, RALFSConfig
from .data import run_preprocessing, build_index
from .retriever import create_retriever
from .generator import create_generator
from .training import train, FiDDataset
from .evaluation import evaluate

__version__ = "1.0.0"
__all__ = [
    "get_logger",
    "RALFSConfig",
    "run_preprocessing",
    "build_index",
    "create_retriever",
    "create_generator",
    "train",
    "evaluate",
]

