# ============================================================================
# File: ralfs/evaluation/human.py
# ============================================================================
"""
Human evaluation template generator for RALFS.

Generates evaluation forms for human assessors to rate summaries on:
- Relevance
- Coherence
- Consistency (faithfulness)
- Fluency
"""

from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
import random
from ralfs.core.logging import get_logger
from ralfs.utils.io import load_json, load_jsonl

logger = get_logger(__name__)


def create_human_eval_template(
    predictions: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
    output_path: Path | str,
    num_samples: int = 50,
    randomize: bool = True,
    seed: int = 42,
) -> Path:
    """
    Create human evaluation template (CSV format).
    
    Args:
        predictions: List of predictions with 'id' and 'summary'
        references: List of references with 'id' and 'summary'
        output_path: Output CSV path
        num_samples: Number of samples to include
        randomize: Whether to randomize sample selection
        seed: Random seed
    
    Returns:
        Path to created CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Match predictions with references
    pred_dict = {p['id']: p for p in predictions}
    ref_dict = {r['id']: r for r in references}
    
    common_ids = list(set(pred_dict.keys()) & set(ref_dict.keys()))
    
    if not common_ids:
        raise ValueError("No matching IDs between predictions and references")
    
    # Sample
    if randomize:
        random.seed(seed)
        random.shuffle(common_ids)
    
    sample_ids = common_ids[:num_samples]
    
    logger.info(f"Creating human evaluation template for {len(sample_ids)} samples")
    
    # Create CSV
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'ID',
            'Reference Summary',
            'Generated Summary',
            'Relevance (1-5)',
            'Coherence (1-5)',
            'Consistency (1-5)',
            'Fluency (1-5)',
            'Overall (1-5)',
            'Comments',
        ])
        
        # Rows
        for sample_id in sample_ids:
            ref_summary = ref_dict[sample_id]['summary']
            pred_summary = pred_dict[sample_id]['summary']
            
            writer.writerow([
                sample_id,
                ref_summary,
                pred_summary,
                '',  # Relevance
                '',  # Coherence
                '',  # Consistency
                '',  # Fluency
                '',  # Overall
                '',  # Comments
            ])
    
    logger.info(f"Human evaluation template saved: {output_path}")
    logger.info("\nEvaluation Guidelines:")
    logger.info("  Relevance (1-5): How relevant is the summary to the source?")
    logger.info("  Coherence (1-5): How well-structured and logical is the summary?")
    logger.info("  Consistency (1-5): How faithful is the summary to the source (no hallucinations)?")
    logger.info("  Fluency (1-5): How grammatically correct and readable is the summary?")
    logger.info("  Overall (1-5): Overall quality of the summary")
    
    return output_path


def create_pairwise_comparison_template(
    predictions_a: List[Dict[str, Any]],
    predictions_b: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
    output_path: Path | str,
    num_samples: int = 50,
    system_a_name: str = "System A",
    system_b_name: str = "System B",
    randomize_order: bool = True,
    seed: int = 42,
) -> Path:
    """
    Create pairwise comparison template for A/B testing.
    
    Args:
        predictions_a: System A predictions
        predictions_b: System B predictions
        references: Ground truth references
        output_path: Output CSV path
        num_samples: Number of samples
        system_a_name: Name for system A
        system_b_name: Name for system B
        randomize_order: Randomize which system appears first
        seed: Random seed
    
    Returns:
        Path to created CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Match all three
    pred_a_dict = {p['id']: p for p in predictions_a}
    pred_b_dict = {p['id']: p for p in predictions_b}
    ref_dict = {r['id']: r for r in references}
    
    common_ids = list(
        set(pred_a_dict.keys()) & set(pred_b_dict.keys()) & set(ref_dict.keys())
    )
    
    if not common_ids:
        raise ValueError("No matching IDs across all inputs")
    
    # Sample
    if randomize_order:
        random.seed(seed)
        random.shuffle(common_ids)
    
    sample_ids = common_ids[:num_samples]
    
    logger.info(f"Creating pairwise comparison for {len(sample_ids)} samples")
    
    # Create CSV
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'ID',
            'Reference Summary',
            'Summary 1',
            'Summary 2',
            'System 1 Name',
            'System 2 Name',
            'Which is Better? (1/2/Tie)',
            'Comments',
        ])
        
        # Rows
        random.seed(seed + 1)  # Different seed for order randomization
        
        for sample_id in sample_ids:
            ref_summary = ref_dict[sample_id]['summary']
            summary_a = pred_a_dict[sample_id]['summary']
            summary_b = pred_b_dict[sample_id]['summary']
            
            # Randomly swap order (blind evaluation)
            if randomize_order and random.random() > 0.5:
                summary_1, summary_2 = summary_b, summary_a
                system_1, system_2 = system_b_name, system_a_name
            else:
                summary_1, summary_2 = summary_a, summary_b
                system_1, system_2 = system_a_name, system_b_name
            
            writer.writerow([
                sample_id,
                ref_summary,
                summary_1,
                summary_2,
                system_1,
                system_2,
                '',  # Preference
                '',  # Comments
            ])
    
    logger.info(f"Pairwise comparison template saved: {output_path}")
    
    return output_path