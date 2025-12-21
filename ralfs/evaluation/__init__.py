# ============================================================================
# File: ralfs/evaluation/__init__.py
# ============================================================================
"""Evaluation module with ROUGE, BERTScore, and EGF."""

from .faithfulness import EntityGrid, EntityGridFaithfulness, compute_egf
from .human import create_human_eval_template
from .main import (
    AggregatedResults,
    EvaluationResult,
    RALFSEvaluator,
    compare_systems,
    run_evaluation,
)
from .metrics import evaluate_bertscore, evaluate_predictions, evaluate_rouge

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
