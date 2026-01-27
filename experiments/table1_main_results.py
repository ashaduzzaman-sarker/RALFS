#!/usr/bin/env python3
"""
Reproduce Table 1: Main Results (ArXiv & GovReport)

Paper claim: RALFS achieves +12% ROUGE-2 over fixed-k FiD with 60% token reduction.

Usage:
    python -m experiments.table1_main_results --datasets arxiv govreport --seeds 13 42 2024
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json
import pandas as pd
from omegaconf import OmegaConf

from ralfs import (
    load_config,
    setup_logging,
    get_logger,
    set_seed,
    create_retriever,
    create_generator,
    evaluate_rouge,
    evaluate_bertscore,
    compute_egf,
)
from ralfs.utils import ExperimentTracker
from ralfs.training.trainer import train_model
from ralfs.evaluation.main import run_evaluation

logger = get_logger(__name__)


def run_system(
    system_name: str,
    dataset: str,
    seed: int,
    adaptive_k: bool = True,
    k_fixed: int = 20,
    output_dir: Path = Path("results/main_results"),
) -> Dict[str, Any]:
    """
    Run a complete system (RALFS or baseline).
    
    Args:
        system_name: Name of system (e.g., "RALFS", "Fixed-k-20")
        dataset: Dataset name (arxiv or govreport)
        seed: Random seed
        adaptive_k: Whether to use adaptive k selection
        k_fixed: Fixed k value if adaptive_k=False
        output_dir: Output directory
        
    Returns:
        Dictionary of metrics
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Running: {system_name} on {dataset} (seed={seed})")
    logger.info(f"{'='*70}")
    
    # Set seed for reproducibility
    set_seed(seed, deterministic=True)
    
    # Load config
    cfg = load_config()
    cfg.data.dataset = dataset
    cfg.data.seed = seed
    cfg.generator.adaptive_k = adaptive_k
    if not adaptive_k:
        cfg.retriever.k_final = k_fixed
        cfg.generator.adaptive_k_min = k_fixed
        cfg.generator.adaptive_k_max = k_fixed
    
    # Create experiment tracker
    exp_name = f"{system_name}_{dataset}_seed{seed}"
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    tracker = ExperimentTracker(exp_dir, seed=seed)
    tracker.log_config(OmegaConf.to_container(cfg, resolve=True))
    
    try:
        # Train model
        logger.info(f"Training {system_name}...")
        train_stats = train_model(cfg)
        tracker.log_metrics({"train_loss": train_stats.get("final_loss", 0.0)})
        
        # Evaluate
        logger.info(f"Evaluating {system_name}...")
        predictions_path = exp_dir / "predictions.json"
        references_path = Path("data/processed") / f"{dataset}_test.json"
        
        eval_results = run_evaluation(
            predictions_path,
            references_path,
            exp_dir / "evaluation",
            metrics=['rouge', 'bertscore', 'egf'],
        )
        
        # Collect metrics
        metrics = {
            'system': system_name,
            'dataset': dataset,
            'seed': seed,
            'rouge1': eval_results['rouge1_mean'],
            'rouge2': eval_results['rouge2_mean'],
            'rougeL': eval_results['rougeL_mean'],
            'bertscore_f1': eval_results['bertscore_f1_mean'],
            'egf': eval_results['egf_mean'],
            'tokens_mean': eval_results.get('tokens_mean', 0),
            'k_mean': eval_results.get('k_mean', k_fixed if not adaptive_k else 0),
        }
        
        tracker.log_metrics(metrics)
        tracker.save(notes=f"Table 1: {system_name} on {dataset}")
        
        logger.info(f"✓ {system_name} completed:")
        logger.info(f"  ROUGE-2: {metrics['rouge2']:.4f}")
        logger.info(f"  Tokens: {metrics['tokens_mean']:.0f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"✗ {system_name} failed: {e}", exc_info=True)
        return {'system': system_name, 'dataset': dataset, 'seed': seed, 'error': str(e)}


def compute_statistics(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Compute mean ± std across seeds with 95% CI.
    
    Args:
        results: List of per-seed results
        
    Returns:
        DataFrame with aggregated statistics
    """
    df = pd.DataFrame(results)
    
    # Group by system and dataset
    grouped = df.groupby(['system', 'dataset'])
    
    # Compute statistics
    stats = []
    for (system, dataset), group in grouped:
        n = len(group)
        
        # Mean ± std for each metric
        row = {
            'System': system,
            'Dataset': dataset,
            'N': n,
        }
        
        for metric in ['rouge1', 'rouge2', 'rougeL', 'bertscore_f1', 'egf', 'tokens_mean', 'k_mean']:
            if metric in group.columns:
                mean = group[metric].mean()
                std = group[metric].std()
                # 95% CI: mean ± 1.96 * std / sqrt(n)
                ci = 1.96 * std / (n ** 0.5) if n > 1 else 0.0
                row[f'{metric}_mean'] = mean
                row[f'{metric}_ci'] = ci
        
        stats.append(row)
    
    return pd.DataFrame(stats)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Reproduce Table 1: Main Results")
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['arxiv', 'govreport'],
        choices=['arxiv', 'govreport'],
        help="Datasets to evaluate"
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=[13, 42, 2024],
        help="Random seeds for averaging"
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/main_results'),
        help="Output directory"
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help="Skip training (use existing checkpoints)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger.info("=" * 70)
    logger.info("RALFS Table 1: Main Results Reproduction")
    logger.info("=" * 70)
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Output: {args.output}")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    all_results = []
    
    for dataset in args.datasets:
        for seed in args.seeds:
            # RALFS (adaptive k)
            result_ralfs = run_system(
                system_name="RALFS",
                dataset=dataset,
                seed=seed,
                adaptive_k=True,
                output_dir=args.output,
            )
            all_results.append(result_ralfs)
            
            # Fixed-k baselines
            for k in [5, 10, 15, 20]:
                result_fixed = run_system(
                    system_name=f"Fixed-k-{k}",
                    dataset=dataset,
                    seed=seed,
                    adaptive_k=False,
                    k_fixed=k,
                    output_dir=args.output,
                )
                all_results.append(result_fixed)
    
    # Save raw results
    raw_results_path = args.output / "table1_raw_results.json"
    with open(raw_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Raw results saved to: {raw_results_path}")
    
    # Compute statistics
    stats_df = compute_statistics(all_results)
    
    # Format table for paper
    table1_path = args.output / "table1.csv"
    stats_df.to_csv(table1_path, index=False)
    logger.info(f"Table 1 saved to: {table1_path}")
    
    # Print table
    logger.info("\n" + "=" * 70)
    logger.info("TABLE 1: Main Results (Mean ± 95% CI)")
    logger.info("=" * 70)
    print(stats_df.to_string(index=False))
    
    # Print improvement over Fixed-k-20
    logger.info("\n" + "=" * 70)
    logger.info("Improvements vs. Fixed-k-20 Baseline")
    logger.info("=" * 70)
    
    for dataset in args.datasets:
        ralfs_row = stats_df[(stats_df['System'] == 'RALFS') & (stats_df['Dataset'] == dataset)]
        baseline_row = stats_df[(stats_df['System'] == 'Fixed-k-20') & (stats_df['Dataset'] == dataset)]
        
        if not ralfs_row.empty and not baseline_row.empty:
            rouge2_ralfs = ralfs_row['rouge2_mean'].values[0]
            rouge2_baseline = baseline_row['rouge2_mean'].values[0]
            tokens_ralfs = ralfs_row['tokens_mean_mean'].values[0]
            tokens_baseline = baseline_row['tokens_mean_mean'].values[0]
            
            rouge2_improvement = ((rouge2_ralfs - rouge2_baseline) / rouge2_baseline) * 100
            tokens_reduction = ((tokens_baseline - tokens_ralfs) / tokens_baseline) * 100
            
            logger.info(f"{dataset.upper()}:")
            logger.info(f"  ROUGE-2: {rouge2_ralfs:.4f} vs {rouge2_baseline:.4f} (+{rouge2_improvement:.1f}%)")
            logger.info(f"  Tokens:  {tokens_ralfs:.0f} vs {tokens_baseline:.0f} (-{tokens_reduction:.1f}%)")
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ Table 1 reproduction complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
