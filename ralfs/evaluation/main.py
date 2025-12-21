# ============================================================================
# File: ralfs/evaluation/main.py
# ============================================================================
"""
Complete evaluation pipeline for RALFS.

Evaluates generated summaries with ROUGE, BERTScore, and EGF metrics.
Supports batch evaluation, statistical analysis (bootstrap confidence intervals,
paired t-tests), and export to multiple formats suitable for conference papers.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from tqdm.auto import tqdm

from ralfs.core.logging import get_logger
from ralfs.evaluation.faithfulness import compute_egf
from ralfs.utils.io import load_json, load_jsonl, save_json

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Single evaluation result."""

    id: str
    reference: str
    prediction: str
    rouge1: float
    rouge2: float
    rougeL: float
    bertscore_f1: float
    egf: float
    metadata: dict[str, Any] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AggregatedResults:
    """Aggregated evaluation results with confidence intervals."""

    num_samples: int
    rouge1_mean: float
    rouge1_std: float
    rouge1_ci_lower: float
    rouge1_ci_upper: float
    rouge2_mean: float
    rouge2_std: float
    rouge2_ci_lower: float
    rouge2_ci_upper: float
    rougeL_mean: float
    rougeL_std: float
    rougeL_ci_lower: float
    rougeL_ci_upper: float
    bertscore_f1_mean: float
    bertscore_f1_std: float
    bertscore_f1_ci_lower: float
    bertscore_f1_ci_upper: float
    egf_mean: float
    egf_std: float
    egf_ci_lower: float
    egf_ci_upper: float
    evaluation_time: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """Pretty print results with confidence intervals."""
        lines = [
            "=" * 70,
            "EVALUATION RESULTS (95% Confidence Intervals)",
            "=" * 70,
            f"Samples: {self.num_samples}",
            f"Evaluation Time: {self.evaluation_time:.2f}s",
            "",
            "ROUGE Scores:",
            f"  ROUGE-1:  {self.rouge1_mean:.4f} ± {self.rouge1_std:.4f}  "
            f"[{self.rouge1_ci_lower:.4f}, {self.rouge1_ci_upper:.4f}]",
            f"  ROUGE-2:  {self.rouge2_mean:.4f} ± {self.rouge2_std:.4f}  "
            f"[{self.rouge2_ci_lower:.4f}, {self.rouge2_ci_upper:.4f}]",
            f"  ROUGE-L:  {self.rougeL_mean:.4f} ± {self.rougeL_std:.4f}  "
            f"[{self.rougeL_ci_lower:.4f}, {self.rougeL_ci_upper:.4f}]",
            "",
            "BERTScore:",
            f"  F1:       {self.bertscore_f1_mean:.4f} ± {self.bertscore_f1_std:.4f}  "
            f"[{self.bertscore_f1_ci_lower:.4f}, {self.bertscore_f1_ci_upper:.4f}]",
            "",
            "Entity Grid Faithfulness (EGF):",
            f"  EGF:      {self.egf_mean:.4f} ± {self.egf_std:.4f}  "
            f"[{self.egf_ci_lower:.4f}, {self.egf_ci_upper:.4f}]",
            "=" * 70,
        ]
        return "\n".join(lines)


class RALFSEvaluator:
    """
    Complete evaluator for RALFS summaries.

    Features:
    - ROUGE (1, 2, L)
    - BERTScore
    - EGF (Entity Grid Faithfulness)
    - Statistical analysis (mean, std, confidence intervals)
    - Export to JSON, CSV, LaTeX
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        lang: str = "en",
        verbose: bool = True,
    ):
        """
        Initialize evaluator.

        Args:
            metrics: List of metrics to compute (default: all)
            lang: Language for BERTScore
            verbose: Show progress bars
        """
        self.metrics = metrics or ["rouge", "bertscore", "egf"]
        self.lang = lang
        self.verbose = verbose

        logger.info(f"Initialized evaluator with metrics: {self.metrics}")

    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        references: list[dict[str, Any]],
    ) -> tuple[list[EvaluationResult], AggregatedResults]:
        """
        Evaluate predictions against references.

        Args:
            predictions: List of dicts with 'id' and 'summary' keys
            references: List of dicts with 'id' and 'summary' keys

        Returns:
            Tuple of (individual results, aggregated results)
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs "
                f"{len(references)} references"
            )

        logger.info(f"Evaluating {len(predictions)} samples...")
        start_time = time.time()

        # Extract texts
        pred_texts = [p["summary"] for p in predictions]
        ref_texts = [r["summary"] for r in references]

        # Compute metrics
        results = []

        # Individual metrics (for per-sample results)
        logger.info("Computing individual metrics...")

        for i, (pred, ref) in enumerate(
            tqdm(
                zip(predictions, references, strict=False),
                total=len(predictions),
                desc="Evaluating",
                disable=not self.verbose,
            )
        ):
            pred_id = pred.get("id", f"sample_{i}")
            pred_text = pred["summary"]
            ref_text = ref["summary"]

            # ROUGE
            if "rouge" in self.metrics:
                from rouge_score import rouge_scorer

                scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
                rouge_scores = scorer.score(ref_text, pred_text)
                rouge1 = rouge_scores["rouge1"].fmeasure
                rouge2 = rouge_scores["rouge2"].fmeasure
                rougeL = rouge_scores["rougeL"].fmeasure
            else:
                rouge1 = rouge2 = rougeL = 0.0

            # BERTScore (batch computed below)
            bertscore_f1 = 0.0  # Placeholder

            # EGF
            if "egf" in self.metrics:
                egf = compute_egf(ref_text, pred_text)
            else:
                egf = 0.0

            results.append(
                EvaluationResult(
                    id=pred_id,
                    reference=ref_text,
                    prediction=pred_text,
                    rouge1=rouge1,
                    rouge2=rouge2,
                    rougeL=rougeL,
                    bertscore_f1=bertscore_f1,
                    egf=egf,
                    metadata=pred.get("metadata", {}),
                )
            )

        # Batch BERTScore (more efficient)
        if "bertscore" in self.metrics:
            logger.info("Computing BERTScore (batch)...")
            from bert_score import score as bert_score_fn

            _, _, F1 = bert_score_fn(
                pred_texts,
                ref_texts,
                lang=self.lang,
                verbose=self.verbose,
            )

            for i, f1 in enumerate(F1.tolist()):
                results[i].bertscore_f1 = f1

        # Aggregate results
        aggregated = self._aggregate_results(results, time.time() - start_time)

        logger.info("Evaluation complete!")
        logger.info(f"\n{aggregated}")

        return results, aggregated

    def _aggregate_results(
        self,
        results: list[EvaluationResult],
        evaluation_time: float,
    ) -> AggregatedResults:
        """
        Compute aggregated statistics with bootstrap confidence intervals.

        Uses bootstrap resampling (1000 iterations) to compute 95% confidence
        intervals for all metrics, providing robust uncertainty estimates for
        conference paper reporting.
        """
        rouge1_scores = [r.rouge1 for r in results]
        rouge2_scores = [r.rouge2 for r in results]
        rougeL_scores = [r.rougeL for r in results]
        bertscore_scores = [r.bertscore_f1 for r in results]
        egf_scores = [r.egf for r in results]

        # Compute bootstrap confidence intervals
        rouge1_ci = self._bootstrap_ci(rouge1_scores)
        rouge2_ci = self._bootstrap_ci(rouge2_scores)
        rougeL_ci = self._bootstrap_ci(rougeL_scores)
        bertscore_ci = self._bootstrap_ci(bertscore_scores)
        egf_ci = self._bootstrap_ci(egf_scores)

        return AggregatedResults(
            num_samples=len(results),
            rouge1_mean=np.mean(rouge1_scores),
            rouge1_std=np.std(rouge1_scores),
            rouge1_ci_lower=rouge1_ci[0],
            rouge1_ci_upper=rouge1_ci[1],
            rouge2_mean=np.mean(rouge2_scores),
            rouge2_std=np.std(rouge2_scores),
            rouge2_ci_lower=rouge2_ci[0],
            rouge2_ci_upper=rouge2_ci[1],
            rougeL_mean=np.mean(rougeL_scores),
            rougeL_std=np.std(rougeL_scores),
            rougeL_ci_lower=rougeL_ci[0],
            rougeL_ci_upper=rougeL_ci[1],
            bertscore_f1_mean=np.mean(bertscore_scores),
            bertscore_f1_std=np.std(bertscore_scores),
            bertscore_f1_ci_lower=bertscore_ci[0],
            bertscore_f1_ci_upper=bertscore_ci[1],
            egf_mean=np.mean(egf_scores),
            egf_std=np.std(egf_scores),
            egf_ci_lower=egf_ci[0],
            egf_ci_upper=egf_ci[1],
            evaluation_time=evaluation_time,
        )

    def _bootstrap_ci(
        self,
        scores: list[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """
        Compute bootstrap confidence interval.

        Args:
            scores: List of metric scores
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (default: 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(scores) < 2:
            mean = np.mean(scores) if scores else 0.0
            return (mean, mean)

        bootstrap_means = []
        n = len(scores)

        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(scores, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        # Compute percentiles
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        return (lower, upper)

    def save_results(
        self,
        results: list[EvaluationResult],
        aggregated: AggregatedResults,
        output_dir: Path | str,
        prefix: str = "eval",
    ):
        """
        Save evaluation results to multiple formats.

        Args:
            results: Individual results
            aggregated: Aggregated results
            output_dir: Output directory
            prefix: Filename prefix
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual results (JSON)
        individual_path = output_dir / f"{prefix}_individual.json"
        save_json([r.to_dict() for r in results], individual_path)
        logger.info(f"Saved individual results: {individual_path}")

        # Save aggregated results (JSON)
        aggregated_path = output_dir / f"{prefix}_aggregated.json"
        save_json(aggregated.to_dict(), aggregated_path)
        logger.info(f"Saved aggregated results: {aggregated_path}")

        # Save CSV
        self._save_csv(results, output_dir / f"{prefix}_results.csv")

        # Save LaTeX table
        self._save_latex(aggregated, output_dir / f"{prefix}_latex.tex")

    def _save_csv(self, results: list[EvaluationResult], path: Path):
        """Save results as CSV."""
        import csv

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "id",
                    "rouge1",
                    "rouge2",
                    "rougeL",
                    "bertscore_f1",
                    "egf",
                    "reference",
                    "prediction",
                ]
            )

            # Rows
            for r in results:
                writer.writerow(
                    [
                        r.id,
                        r.rouge1,
                        r.rouge2,
                        r.rougeL,
                        r.bertscore_f1,
                        r.egf,
                        r.reference[:100],
                        r.prediction[:100],
                    ]
                )

        logger.info(f"Saved CSV: {path}")

    def _save_latex(self, aggregated: AggregatedResults, path: Path):
        """Save aggregated results as LaTeX table with confidence intervals."""
        latex = f"""\\begin{{table}}[t]
\\centering
\\caption{{RALFS Evaluation Results (n={aggregated.num_samples})}}
\\label{{tab:ralfs-results}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Mean ± Std}} & \\textbf{{95\\% CI}} \\\\
\\midrule
ROUGE-1 & {aggregated.rouge1_mean:.4f} $\\pm$ {aggregated.rouge1_std:.4f} & [{aggregated.rouge1_ci_lower:.4f}, {aggregated.rouge1_ci_upper:.4f}] \\\\
ROUGE-2 & {aggregated.rouge2_mean:.4f} $\\pm$ {aggregated.rouge2_std:.4f} & [{aggregated.rouge2_ci_lower:.4f}, {aggregated.rouge2_ci_upper:.4f}] \\\\
ROUGE-L & {aggregated.rougeL_mean:.4f} $\\pm$ {aggregated.rougeL_std:.4f} & [{aggregated.rougeL_ci_lower:.4f}, {aggregated.rougeL_ci_upper:.4f}] \\\\
BERTScore-F1 & {aggregated.bertscore_f1_mean:.4f} $\\pm$ {aggregated.bertscore_f1_std:.4f} & [{aggregated.bertscore_f1_ci_lower:.4f}, {aggregated.bertscore_f1_ci_upper:.4f}] \\\\
EGF & {aggregated.egf_mean:.4f} $\\pm$ {aggregated.egf_std:.4f} & [{aggregated.egf_ci_lower:.4f}, {aggregated.egf_ci_upper:.4f}] \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(latex)

        logger.info(f"Saved LaTeX table: {path}")


def compare_systems(
    system1_results: list[EvaluationResult],
    system2_results: list[EvaluationResult],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Compare two systems using paired statistical tests.

    Performs paired t-test and bootstrap significance test to determine
    if differences between systems are statistically significant.

    Args:
        system1_results: Results from first system
        system2_results: Results from second system
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary with test results and p-values

    Example:
        >>> baseline_results = evaluate_system("baseline", test_data)
        >>> ralfs_results = evaluate_system("ralfs", test_data)
        >>> comparison = compare_systems(baseline_results, ralfs_results)
        >>> print(f"ROUGE-L improvement: {comparison['rougeL_diff_mean']:.4f}")
        >>> print(f"Significant: {comparison['rougeL_significant']}")
    """
    if len(system1_results) != len(system2_results):
        raise ValueError("Systems must have same number of samples")

    # Extract scores
    sys1_r1 = [r.rouge1 for r in system1_results]
    sys1_r2 = [r.rouge2 for r in system1_results]
    sys1_rL = [r.rougeL for r in system1_results]
    sys1_bert = [r.bertscore_f1 for r in system1_results]
    sys1_egf = [r.egf for r in system1_results]

    sys2_r1 = [r.rouge1 for r in system2_results]
    sys2_r2 = [r.rouge2 for r in system2_results]
    sys2_rL = [r.rougeL for r in system2_results]
    sys2_bert = [r.bertscore_f1 for r in system2_results]
    sys2_egf = [r.egf for r in system2_results]

    def paired_test(scores1, scores2, metric_name):
        """Perform paired t-test and compute difference statistics."""
        diffs = np.array(scores2) - np.array(scores1)

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores2, scores1)

        # Effect size (Cohen's d)
        cohens_d = np.mean(diffs) / (np.std(diffs) + 1e-10)

        # Bootstrap test
        n_bootstrap = 10000
        bootstrap_diffs = []
        n = len(scores1)

        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            sample1 = [scores1[i] for i in indices]
            sample2 = [scores2[i] for i in indices]
            bootstrap_diffs.append(np.mean(sample2) - np.mean(sample1))

        # Bootstrap p-value (two-tailed)
        bootstrap_p = 2 * min(
            np.mean(np.array(bootstrap_diffs) <= 0), np.mean(np.array(bootstrap_diffs) >= 0)
        )

        return {
            f"{metric_name}_diff_mean": np.mean(diffs),
            f"{metric_name}_diff_std": np.std(diffs),
            f"{metric_name}_t_statistic": t_stat,
            f"{metric_name}_p_value": p_value,
            f"{metric_name}_bootstrap_p_value": bootstrap_p,
            f"{metric_name}_cohens_d": cohens_d,
            f"{metric_name}_significant": p_value < alpha,
            f"{metric_name}_significant_bootstrap": bootstrap_p < alpha,
        }

    # Run tests for all metrics
    results = {}
    results.update(paired_test(sys1_r1, sys2_r1, "rouge1"))
    results.update(paired_test(sys1_r2, sys2_r2, "rouge2"))
    results.update(paired_test(sys1_rL, sys2_rL, "rougeL"))
    results.update(paired_test(sys1_bert, sys2_bert, "bertscore"))
    results.update(paired_test(sys1_egf, sys2_egf, "egf"))

    # Summary
    results["n_samples"] = len(system1_results)
    results["alpha"] = alpha

    logger.info("Statistical Significance Test Results:")
    logger.info(
        f"  ROUGE-L: Δ={results['rougeL_diff_mean']:.4f}, "
        f"p={results['rougeL_p_value']:.4f}, "
        f"sig={'Yes' if results['rougeL_significant'] else 'No'}"
    )
    logger.info(
        f"  ROUGE-2: Δ={results['rouge2_diff_mean']:.4f}, "
        f"p={results['rouge2_p_value']:.4f}, "
        f"sig={'Yes' if results['rouge2_significant'] else 'No'}"
    )

    return results


def run_evaluation(
    predictions_path: Path | str,
    references_path: Path | str,
    output_dir: Path | str,
    metrics: list[str] | None = None,
) -> AggregatedResults:
    """
    Run complete evaluation pipeline.

    Args:
        predictions_path: Path to predictions JSON/JSONL
        references_path: Path to references JSON/JSONL
        output_dir: Output directory for results
        metrics: List of metrics (default: all)

    Returns:
        Aggregated results

    Example:
        >>> results = run_evaluation(
        ...     "results/predictions.json",
        ...     "data/test/references.json",
        ...     "results/evaluation"
        ... )
    """
    # Load data
    predictions_path = Path(predictions_path)
    references_path = Path(references_path)

    logger.info(f"Loading predictions from {predictions_path}")
    if predictions_path.suffix == ".jsonl":
        predictions = load_jsonl(predictions_path)
    else:
        predictions = load_json(predictions_path)

    logger.info(f"Loading references from {references_path}")
    if references_path.suffix == ".jsonl":
        references = load_jsonl(references_path)
    else:
        references = load_json(references_path)

    # Evaluate
    evaluator = RALFSEvaluator(metrics=metrics)
    individual_results, aggregated_results = evaluator.evaluate(predictions, references)

    # Save results
    evaluator.save_results(individual_results, aggregated_results, output_dir)

    return aggregated_results
