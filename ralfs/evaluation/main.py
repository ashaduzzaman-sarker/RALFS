# ============================================================================
# File: ralfs/evaluation/main.py
# ============================================================================
"""
Complete evaluation pipeline for RALFS.

Evaluates generated summaries with ROUGE, BERTScore, and EGF metrics.
Supports batch evaluation, statistical analysis, and export to multiple formats.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
import numpy as np
from tqdm.auto import tqdm

from ralfs.core.logging import get_logger
from ralfs.utils.io import load_json, save_json, load_jsonl
from ralfs.evaluation.metrics import evaluate_rouge, evaluate_bertscore
from ralfs.evaluation.faithfulness import compute_egf

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
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AggregatedResults:
    """Aggregated evaluation results."""
    num_samples: int
    rouge1_mean: float
    rouge1_std: float
    rouge2_mean: float
    rouge2_std: float
    rougeL_mean: float
    rougeL_std: float
    bertscore_f1_mean: float
    bertscore_f1_std: float
    egf_mean: float
    egf_std: float
    evaluation_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def __str__(self) -> str:
        """Pretty print results."""
        lines = [
            "=" * 60,
            "EVALUATION RESULTS",
            "=" * 60,
            f"Samples: {self.num_samples}",
            f"Evaluation Time: {self.evaluation_time:.2f}s",
            "",
            "ROUGE Scores:",
            f"  ROUGE-1:  {self.rouge1_mean:.4f} ± {self.rouge1_std:.4f}",
            f"  ROUGE-2:  {self.rouge2_mean:.4f} ± {self.rouge2_std:.4f}",
            f"  ROUGE-L:  {self.rougeL_mean:.4f} ± {self.rougeL_std:.4f}",
            "",
            "BERTScore:",
            f"  F1:       {self.bertscore_f1_mean:.4f} ± {self.bertscore_f1_std:.4f}",
            "",
            "Entity Grid Faithfulness (EGF):",
            f"  EGF:      {self.egf_mean:.4f} ± {self.egf_std:.4f}",
            "=" * 60,
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
        metrics: Optional[List[str]] = None,
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
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
    ) -> tuple[List[EvaluationResult], AggregatedResults]:
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
        pred_texts = [p['summary'] for p in predictions]
        ref_texts = [r['summary'] for r in references]
        
        # Compute metrics
        results = []
        
        # Individual metrics (for per-sample results)
        logger.info("Computing individual metrics...")
        
        for i, (pred, ref) in enumerate(tqdm(
            zip(predictions, references),
            total=len(predictions),
            desc="Evaluating",
            disable=not self.verbose,
        )):
            pred_id = pred.get('id', f'sample_{i}')
            pred_text = pred['summary']
            ref_text = ref['summary']
            
            # ROUGE
            if "rouge" in self.metrics:
                from rouge_score import rouge_scorer
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                rouge_scores = scorer.score(ref_text, pred_text)
                rouge1 = rouge_scores['rouge1'].fmeasure
                rouge2 = rouge_scores['rouge2'].fmeasure
                rougeL = rouge_scores['rougeL'].fmeasure
            else:
                rouge1 = rouge2 = rougeL = 0.0
            
            # BERTScore (batch computed below)
            bertscore_f1 = 0.0  # Placeholder
            
            # EGF
            if "egf" in self.metrics:
                egf = compute_egf(ref_text, pred_text)
            else:
                egf = 0.0
            
            results.append(EvaluationResult(
                id=pred_id,
                reference=ref_text,
                prediction=pred_text,
                rouge1=rouge1,
                rouge2=rouge2,
                rougeL=rougeL,
                bertscore_f1=bertscore_f1,
                egf=egf,
                metadata=pred.get('metadata', {}),
            ))
        
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
        results: List[EvaluationResult],
        evaluation_time: float,
    ) -> AggregatedResults:
        """Compute aggregated statistics."""
        rouge1_scores = [r.rouge1 for r in results]
        rouge2_scores = [r.rouge2 for r in results]
        rougeL_scores = [r.rougeL for r in results]
        bertscore_scores = [r.bertscore_f1 for r in results]
        egf_scores = [r.egf for r in results]
        
        return AggregatedResults(
            num_samples=len(results),
            rouge1_mean=np.mean(rouge1_scores),
            rouge1_std=np.std(rouge1_scores),
            rouge2_mean=np.mean(rouge2_scores),
            rouge2_std=np.std(rouge2_scores),
            rougeL_mean=np.mean(rougeL_scores),
            rougeL_std=np.std(rougeL_scores),
            bertscore_f1_mean=np.mean(bertscore_scores),
            bertscore_f1_std=np.std(bertscore_scores),
            egf_mean=np.mean(egf_scores),
            egf_std=np.std(egf_scores),
            evaluation_time=evaluation_time,
        )
    
    def save_results(
        self,
        results: List[EvaluationResult],
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
    
    def _save_csv(self, results: List[EvaluationResult], path: Path):
        """Save results as CSV."""
        import csv
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'id', 'rouge1', 'rouge2', 'rougeL',
                'bertscore_f1', 'egf', 'reference', 'prediction'
            ])
            
            # Rows
            for r in results:
                writer.writerow([
                    r.id, r.rouge1, r.rouge2, r.rougeL,
                    r.bertscore_f1, r.egf,
                    r.reference[:100], r.prediction[:100]
                ])
        
        logger.info(f"Saved CSV: {path}")
    
    def _save_latex(self, aggregated: AggregatedResults, path: Path):
        """Save aggregated results as LaTeX table."""
        latex = f"""\\begin{{table}}[t]
\\centering
\\caption{{RALFS Evaluation Results (n={aggregated.num_samples})}}
\\label{{tab:results}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Mean}} & \\textbf{{Std}} \\\\
\\midrule
ROUGE-1 & {aggregated.rouge1_mean:.4f} & {aggregated.rouge1_std:.4f} \\\\
ROUGE-2 & {aggregated.rouge2_mean:.4f} & {aggregated.rouge2_std:.4f} \\\\
ROUGE-L & {aggregated.rougeL_mean:.4f} & {aggregated.rougeL_std:.4f} \\\\
BERTScore & {aggregated.bertscore_f1_mean:.4f} & {aggregated.bertscore_f1_std:.4f} \\\\
EGF & {aggregated.egf_mean:.4f} & {aggregated.egf_std:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(latex)
        
        logger.info(f"Saved LaTeX table: {path}")


def run_evaluation(
    predictions_path: Path | str,
    references_path: Path | str,
    output_dir: Path | str,
    metrics: Optional[List[str]] = None,
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
    if predictions_path.suffix == '.jsonl':
        predictions = load_jsonl(predictions_path)
    else:
        predictions = load_json(predictions_path)
    
    logger.info(f"Loading references from {references_path}")
    if references_path.suffix == '.jsonl':
        references = load_jsonl(references_path)
    else:
        references = load_json(references_path)
    
    # Evaluate
    evaluator = RALFSEvaluator(metrics=metrics)
    individual_results, aggregated_results = evaluator.evaluate(predictions, references)
    
    # Save results
    evaluator.save_results(individual_results, aggregated_results, output_dir)
    
    return aggregated_results

