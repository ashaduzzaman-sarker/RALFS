# src/ralfs/__init__.py
from __future__ import annotations

# Core utilities
from .core import (
    get_logger,
    RALFSConfig,
    load_config,
    ROOT_DIR,
    DATA_DIR,
    PROCESSED_DIR,
    INDEX_DIR,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
)

# Data pipeline
from .data import run_preprocessing, build_index

# Retrieval
from .retriever import (
    BaseRetriever,
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    CrossEncoderReranker,
    ColbertRetriever,
    reciprocal_rank_fusion,
    create_retriever,          # from retriever.factory
)

# Generation
from .generator import (
    BaseGenerator,
    FiDGenerator,
    create_generator,
)

# Training
from .training import FiDDataset, train

# Evaluation
from .evaluation import (
    compute_egf,
    create_human_eval_template,
    evaluate_predictions,
    evaluate,
)

# Utils
from .utils import load_json, save_json

__version__ = "1.0.0"

__all__ = [
    # Core
    "get_logger",
    "RALFSConfig",
    "load_config",
    "ROOT_DIR",
    "DATA_DIR",
    "PROCESSED_DIR",
    "INDEX_DIR",
    "CHECKPOINTS_DIR",
    "RESULTS_DIR",

    # Data
    "run_preprocessing",
    "build_index",

    # Retriever
    "BaseRetriever",
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "CrossEncoderReranker",
    "ColbertRetriever",
    "reciprocal_rank_fusion",
    "create_retriever",

    # Generator
    "BaseGenerator",
    "FiDGenerator",
    "create_generator",

    # Training
    "FiDDataset",
    "train",

    # Evaluation
    "compute_egf",
    "create_human_eval_template",
    "evaluate_predictions",
    "evaluate",

    # Utils
    "load_json",
    "save_json",

    # Version
    "__version__",
]
