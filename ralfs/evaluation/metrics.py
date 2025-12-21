# ============================================================================
# File: ralfs/evaluation/metrics.py
# ============================================================================
"""Evaluation metrics: ROUGE, BERTScore."""

from typing import Any

import numpy as np
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer

from ralfs.core.logging import get_logger

logger = get_logger(__name__)


def evaluate_rouge(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """
    Compute ROUGE scores.

    Args:
        predictions: List of predicted summaries
        references: List of reference summaries

    Returns:
        Dict with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references, strict=False):
        score = scorer.score(ref, pred)
        scores["rouge1"].append(score["rouge1"].fmeasure)
        scores["rouge2"].append(score["rouge2"].fmeasure)
        scores["rougeL"].append(score["rougeL"].fmeasure)

    # Average
    avg_scores = {k: np.mean(v) for k, v in scores.items()}

    logger.info("ROUGE Scores:")
    for metric, score in avg_scores.items():
        logger.info(f"  {metric.upper():12} {score:.4f}")

    return avg_scores


def evaluate_bertscore(
    predictions: list[str],
    references: list[str],
    lang: str = "en",
) -> dict[str, float]:
    """
    Compute BERTScore.

    Args:
        predictions: List of predicted summaries
        references: List of reference summaries
        lang: Language code

    Returns:
        Dict with precision, recall, F1
    """
    P, R, F1 = bert_score_fn(
        predictions,
        references,
        lang=lang,
        verbose=False,
    )

    scores = {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }

    logger.info("BERTScore:")
    for metric, score in scores.items():
        logger.info(f"  {metric.upper():20} {score:.4f}")

    return scores


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Evaluate predictions with multiple metrics.

    Args:
        predictions: List of dicts with 'summary' key
        references: List of dicts with 'summary' key

    Returns:
        Dict with all metric scores
    """
    from ralfs.evaluation.faithfulness import compute_egf

    pred_texts = [p["summary"] for p in predictions]
    ref_texts = [r["summary"] for r in references]

    # ROUGE
    rouge_scores = evaluate_rouge(pred_texts, ref_texts)

    # BERTScore
    bert_scores = evaluate_bertscore(pred_texts, ref_texts)

    # EGF (Entity Grid Faithfulness)
    egf_scores = [compute_egf(ref, pred) for ref, pred in zip(ref_texts, pred_texts, strict=False)]
    egf_avg = np.mean(egf_scores)

    logger.info(f"EGF (Entity Grid Faithfulness): {egf_avg:.4f}")

    # Combine all scores
    all_scores = {
        **rouge_scores,
        **bert_scores,
        "egf": egf_avg,
    }

    logger.info("\nFinal Evaluation Results:")
    for metric, score in all_scores.items():
        logger.info(f"  {metric.upper():20} {score:.4f}")

    return all_scores
