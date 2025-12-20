# ============================================================================
# File: ralfs/evaluation/__init__.py
# ============================================================================
"""Evaluation module with ROUGE, BERTScore, and EGF."""

from .metrics import evaluate_predictions, evaluate_rouge, evaluate_bertscore
from .faithfulness import compute_egf, EntityGridFaithfulness, EntityGrid
from .main import (
    run_evaluation,
    RALFSEvaluator,
    EvaluationResult,
    AggregatedResults,
    compare_systems,
)
from .human import create_human_eval_template

__all__ = [
    "evaluate_predictions",
    "evaluate_rouge",
    "evaluate_bertscore",
    "compute_egf",
    "EntityGridFaithfulness",
    "EntityGrid",
    "run_evaluation",
    "RALFSEvaluator",
    "EvaluationResult",
    "AggregatedResults",
    "compare_systems",
    "create_human_eval_template",
]
