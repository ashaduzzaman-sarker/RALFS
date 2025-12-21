# ============================================================================
# File: ralfs/retriever/utils.py
# ============================================================================
"""Utility functions for retrieval."""


import numpy as np

from ralfs.retriever.base import RetrievalResult


def normalize_scores(scores: list[float]) -> list[float]:
    """
    Normalize scores to [0, 1] range using min-max normalization.

    Args:
        scores: List of scores

    Returns:
        Normalized scores
    """
    if not scores:
        return []

    scores_array = np.array(scores)
    min_score = scores_array.min()
    max_score = scores_array.max()

    if max_score == min_score:
        # All scores are the same
        return [1.0] * len(scores)

    normalized = (scores_array - min_score) / (max_score - min_score + 1e-8)
    return normalized.tolist()


def reciprocal_rank_fusion(
    results_list: list[list[RetrievalResult | dict]],
    k: int = 60,
) -> list[RetrievalResult]:
    """
    Combine multiple retrieval results using Reciprocal Rank Fusion (RRF).

    RRF formula: score(d) = sum_r [ 1 / (k + rank_r(d)) ]

    Args:
        results_list: List of result lists from different retrievers
        k: RRF constant (default: 60, as in original paper)

    Returns:
        Fused and ranked results

    Reference:
        Cormack et al. (2009). "Reciprocal Rank Fusion outperforms Condorcet
        and individual Rank Learning Methods"
    """
    if not results_list:
        return []

    # Collect all unique texts with their RRF scores
    rrf_scores: dict[str, float] = {}
    text_to_result: dict[str, RetrievalResult | dict] = {}

    for results in results_list:
        for rank, result in enumerate(results, 1):
            # Extract text
            if isinstance(result, RetrievalResult):
                text = result.text
            elif isinstance(result, dict):
                text = result["text"]
            else:
                continue

            # Compute RRF score
            rrf_score = 1.0 / (k + rank)

            # Accumulate scores
            if text in rrf_scores:
                rrf_scores[text] += rrf_score
            else:
                rrf_scores[text] = rrf_score
                text_to_result[text] = result

    # Sort by RRF score
    sorted_texts = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Convert to RetrievalResult objects
    fused_results = []
    for rank, (text, rrf_score) in enumerate(sorted_texts, 1):
        original_result = text_to_result[text]

        if isinstance(original_result, RetrievalResult):
            result = original_result
            result.score = rrf_score
            result.rank = rank
            if result.metadata is None:
                result.metadata = {}
            result.metadata["fusion"] = "rrf"

        elif isinstance(original_result, dict):
            result = RetrievalResult(
                text=text,
                score=rrf_score,
                rank=rank,
                doc_id=original_result.get("doc_id"),
                chunk_id=original_result.get("chunk_id"),
                metadata={
                    **original_result.get("metadata", {}),
                    "fusion": "rrf",
                },
            )
        else:
            continue

        fused_results.append(result)

    return fused_results
