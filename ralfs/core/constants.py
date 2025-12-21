# ============================================================================
# File: ralfs/core/constants.py
# ============================================================================
"""Constants and default configurations for RALFS."""

from pathlib import Path

# ============================================================================
# Project Paths
# ============================================================================
# Note: This assumes ralfs package is at ROOT_DIR/ralfs/
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"
CONFIGS_DIR = ROOT_DIR / "configs"

# Create directories on import (only if they don't exist)
for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR, INDEX_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Model Defaults
# ============================================================================
# Dense Retrieval (Sentence Transformers)
DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
# Alternative: "sentence-transformers/all-mpnet-base-v2" (better quality, slower)

# ColBERT (Late Interaction)
# Note: Use "colbert-ir/colbertv2.0" if available, else fallback
DEFAULT_COLBERT_MODEL = "colbert-ir/colbertv2.0"

# Cross-Encoder Reranker
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
# Alternative: "cross-encoder/ms-marco-MiniLM-L-6-v2" (faster)

# Generator (Fusion-in-Decoder)
DEFAULT_GENERATOR_MODEL = "google/flan-t5-large"  # 780M params
# Alternatives:
# - "google/flan-t5-base" (250M, for T4 GPU)
# - "google/flan-t5-xl" (3B, for A100)
# - "facebook/bart-large" (406M, alternative architecture)

# ============================================================================
# Retrieval Defaults
# ============================================================================
# Initial retrieval from each method
DEFAULT_K_DENSE = 100
DEFAULT_K_SPARSE = 100
DEFAULT_K_COLBERT = 100

# After RRF fusion
DEFAULT_K_RERANK = 50  # How many to rerank

# Final top-k after reranking
DEFAULT_K_FINAL = 20

# Fusion weights for RRF (Reciprocal Rank Fusion)
DEFAULT_RRF_K = 60  # Constant for RRF formula: 1/(k + rank)

# ============================================================================
# Generation Defaults
# ============================================================================
DEFAULT_MAX_INPUT_LENGTH = 8192  # For FiD context
DEFAULT_MAX_OUTPUT_LENGTH = 512  # Summary length
DEFAULT_NUM_BEAMS = 4
DEFAULT_LENGTH_PENALTY = 1.0

# Adaptive k parameters (for AdaptiveFiD)
ADAPTIVE_K_MIN = 5
ADAPTIVE_K_MAX = 30
ADAPTIVE_K_DEFAULT = 20

# ============================================================================
# Training Defaults
# ============================================================================
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_EPOCHS = 3
DEFAULT_WARMUP_STEPS = 500
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4

# LoRA defaults
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.1
DEFAULT_LORA_TARGET_MODULES = ["q", "v"]  # For T5

# ============================================================================
# Evaluation Defaults
# ============================================================================
ROUGE_TYPES = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"

# ============================================================================
# Dataset Configs
# ============================================================================
SUPPORTED_DATASETS = [
    "arxiv",
    "pubmed",
    "govreport",
    "booksum",
    "multi_news",
]

DATASET_HF_PATHS = {
    "arxiv": "ccdv/arxiv-summarization",
    "pubmed": "ccdv/pubmed-summarization",
    "govreport": "ccdv/govreport-summarization",
    "booksum": "kmfoda/booksum",
    "multi_news": "multi_news",
}

# ============================================================================
# External Dependencies Versions
# ============================================================================
SPACY_MODEL_VERSION = "3.7.1"
SPACY_MODEL_NAME = "en_core_web_sm"
SPACY_MODEL_URL = f"https://github.com/explosion/spacy-models/releases/download/{SPACY_MODEL_NAME}-{SPACY_MODEL_VERSION}/{SPACY_MODEL_NAME}-{SPACY_MODEL_VERSION}.tar.gz"
COLBERT_REPO_URL = "git+https://github.com/stanford-futuredata/ColBERT.git"

# ============================================================================
# Chunking Defaults
# ============================================================================
DEFAULT_CHUNK_SIZE = 512  # tokens
DEFAULT_CHUNK_OVERLAP = 128  # tokens
DEFAULT_MIN_CHUNK_SIZE = 100  # Don't create chunks smaller than this
