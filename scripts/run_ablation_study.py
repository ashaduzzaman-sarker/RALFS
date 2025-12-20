#!/usr/bin/env python3
"""
Ablation study script for RALFS.

Tests the impact of individual components:
1. Adaptive k vs. fixed k
2. LoRA rank (r=8, 16, 32)
3. Number of passages (k=5, 10, 15, 20)
4. Retrieval methods (dense, sparse, hybrid)
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
import json
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ralfs.core.logging import setup_logging, get_logger
from ralfs.core.config import load_config
from ralfs.training.trainer import train_model
from ralfs.evaluation.main import run_evaluation
from ralfs.utils import ExperimentTracker, set_seed

logger = get_logger(__name__)


def run_single_experiment(
    experiment_name: str,
    config_overrides: Dict[str, Any],
    base_config_path: str = "configs/ralfs.yaml",
    seed: int = 42,
) -> Dict[str, float]:
    """
    Run a single experiment with given configuration.
    
    Args:
        experiment_name: Name for this experiment
        config_overrides: Configuration overrides
        base_config_path: Path to base configuration
        seed: Random seed
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"=" * 70)
    logger.info(f"Running experiment: {experiment_name}")
    logger.info(f"Overrides: {config_overrides}")
    logger.info(f"=" * 70)
    
    # Load and override config
    cfg = load_config(base_config_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.create(config_overrides))
    
    # Set seed
    set_seed(seed, deterministic=True)
    
    # Initialize experiment tracker
    output_dir = Path("experiments") / "ablation" / experiment_name
    tracker = ExperimentTracker(output_dir, seed=seed)
    tracker.log_config(config_overrides)
    
    try:
        # Train model
        logger.info("Training model...")
        train_stats = train_model(cfg)
        
        # Evaluate
        logger.info("Evaluating model...")
        predictions_path = output_dir / "predictions.json"
        references_path = Path("data/processed") / f"{cfg.data.dataset}_test.json"
        
        eval_results = run_evaluation(
            predictions_path,
            references_path,
            output_dir / "evaluation",
            metrics=['rouge', 'bertscore', 'egf'],
        )
        
        # Log metrics
        metrics = {
            'rouge1': eval_results.rouge1_mean,
            'rouge2': eval_results.rouge2_mean,
            'rougeL': eval_results.rougeL_mean,
            'bertscore_f1': eval_results.bertscore_f1_mean,
            'egf': eval_results.egf_mean,
        }
        tracker.log_metrics(metrics)
        tracker.save(notes=f"Ablation study: {experiment_name}")
        
        logger.info(f"✓ Experiment '{experiment_name}' completed")
        logger.info(f"  ROUGE-L: {metrics['rougeL']:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"✗ Experiment '{experiment_name}' failed: {e}")
        return {}


def ablation_adaptive_k():
    """Ablation: Adaptive k vs. fixed k."""
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION 1: Adaptive k vs. Fixed k")
    logger.info("=" * 70)
    
    results = {}
    
    # Baseline: Adaptive k
    results['adaptive_k'] = run_single_experiment(
        "adaptive_k",
        {
            'generator': {
                'adaptive': {
                    'enabled': True,
                    'strategy': 'score_dropoff',
                }
            }
        }
    )
    
    # Fixed k values
    for k in [5, 10, 15, 20]:
        results[f'fixed_k_{k}'] = run_single_experiment(
            f"fixed_k_{k}",
            {
                'generator': {
                    'adaptive': {
                        'enabled': False,
                        'default_k': k,
                    }
                }
            }
        )
    
    return results


def ablation_lora_rank():
    """Ablation: LoRA rank."""
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION 2: LoRA Rank")
    logger.info("=" * 70)
    
    results = {}
    
    for r in [8, 16, 32]:
        results[f'lora_r_{r}'] = run_single_experiment(
            f"lora_r_{r}",
            {
                'generator': {
                    'lora': {
                        'r': r,
                        'alpha': 2 * r,  # alpha = 2*r convention
                    }
                }
            }
        )
    
    return results


def ablation_retrieval_method():
    """Ablation: Retrieval methods."""
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION 3: Retrieval Methods")
    logger.info("=" * 70)
    
    results = {}
    
    for method in ['dense', 'sparse', 'hybrid']:
        results[f'retrieval_{method}'] = run_single_experiment(
            f"retrieval_{method}",
            {
                'retriever': {
                    'type': method,
                }
            }
        )
    
    return results


def main():
    """Run complete ablation study."""
    setup_logging()
    
    logger.info("Starting RALFS Ablation Study")
    logger.info("=" * 70)
    
    # Run ablation studies
    all_results = {}
    
    # 1. Adaptive k
    all_results['adaptive_k'] = ablation_adaptive_k()
    
    # 2. LoRA rank
    all_results['lora_rank'] = ablation_lora_rank()
    
    # 3. Retrieval method
    all_results['retrieval_method'] = ablation_retrieval_method()
    
    # Save all results
    results_dir = Path("experiments/ablation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info("\n" + "=" * 70)
    logger.info("Ablation Study Complete")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {results_dir / 'all_results.json'}")


if __name__ == "__main__":
    main()
