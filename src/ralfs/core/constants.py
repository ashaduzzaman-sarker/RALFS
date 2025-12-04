from __future__ import annotations
from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"

# Model defaults
DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
DEFAULT_COLBERT_MODEL = "colbert-ir/colbertv2.0"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
DEFAULT_GENERATOR_MODEL = "google/flan-t5-large"

# Retrieval defaults
DEFAULT_K_DENSE = 100
DEFAULT_K_SPARSE = 100
DEFAULT_K_COLBERT = 100
DEFAULT_K_FINAL = 20