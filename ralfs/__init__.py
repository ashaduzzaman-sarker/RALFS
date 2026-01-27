"""
RALFS: Retrieval-Augmented Long-Form Summarization

A research-grade system for long-document summarization with:
- Adaptive k-selection for Fusion-in-Decoder (novel contribution)
- Entity Grid Faithfulness (EGF) metric (novel contribution)
- Hybrid retrieval (dense + sparse + ColBERT)
- Reproducible experiment infrastructure

Conference Paper: NeurIPS/ACL 2026
"""

from __future__ import annotations

__version__ = "1.0.0"

# Core utilities
from .core import (
    RALFSConfig,
    load_config,
    get_logger,
    setup_logging,
    ROOT_DIR,
    DATA_DIR,
    PROCESSED_DIR,
    INDEX_DIR,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
)

# Data pipeline
from .data import (
    run_preprocessing,
    build_index,
    IndexBuilder,
)

# Retrieval
from .retriever import (
    BaseRetriever,
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    create_retriever,
)

# Generation
from .generator import (
    BaseGenerator,
    FiDGenerator,
    create_generator,
)

# Training
from .training import FiDDataset

# Evaluation
from .evaluation import (
    compute_egf,
    evaluate_rouge,
    evaluate_bertscore,
)

# Utils
from .utils import (
    load_json,
    save_json,
    set_seed,
)

__all__ = [
    # Version
    "__version__",
    
    # Core
    "RALFSConfig",
    "load_config",
    "get_logger",
    "setup_logging",
    "ROOT_DIR",
    "DATA_DIR",
    "PROCESSED_DIR",
    "INDEX_DIR",
    "CHECKPOINTS_DIR",
    "RESULTS_DIR",
    
    # Data
    "run_preprocessing",
    "build_index",
    "IndexBuilder",
    
    # Retriever
    "BaseRetriever",
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "create_retriever",
    
    # Generator
    "BaseGenerator",
    "FiDGenerator",
    "create_generator",
    
    # Training
    "FiDDataset",
    
    # Evaluation
    "compute_egf",
    "evaluate_rouge",
    "evaluate_bertscore",
    
    # Utils
    "load_json",
    "save_json",
    "set_seed",
]
