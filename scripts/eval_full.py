#!/usr/bin/env python3
"""
Full evaluation script: Generates results/experiment_log.csv from trained models.

Evaluates RALFS and baselines on test sets, computing ROUGE, BERTScore, EGF, and token counts.
Saves comprehensive metrics to CSV for comparison and paper figures.
"""

import os
import csv
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

from ralfs.core.logging import get_logger
from ralfs.core.config import load_config
from ralfs.training.dataset import FiDDataset, create_dataloader
from ralfs.evaluation.metrics import evaluate_rouge, evaluate_bertscore
from ralfs.evaluation.faithfulness import compute_egf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

logger = get_logger(__name__)


class EvaluationRunner:
    """Comprehensive evaluation runner for RALFS and baselines."""
    
    def __init__(self, checkpoint_dir: Path = Path("checkpoints")):
        """Initialize evaluation runner."""
        self.checkpoint_dir = checkpoint_dir
        self.results = []
    
    def evaluate_model(
        self,
        model_name: str,
        dataset_name: str,
        test_data_path: Path,
        model_checkpoint: Path,
        tokenizer_checkpoint: Path,
        max_samples: int = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single model on a dataset.
        
        Args:
            model_name: Name of model (e.g., "RALFS", "Standard FiD")
            dataset_name: Dataset name (e.g., "Arxiv", "GovReport")
            test_data_path: Path to test data JSONL
            model_checkpoint: Path to model checkpoint
            tokenizer_checkpoint: Path to tokenizer
            max_samples: Max samples to evaluate (None = all)
        
        Returns:
            Dict with all evaluation metrics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating {model_name} on {dataset_name}")
        logger.info(f"{'='*80}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_checkpoint))
        
        # Load model
        if model_checkpoint.exists():
            model = AutoModelForSeq2SeqLM.from_pretrained(str(model_checkpoint))
            
            # Try to load LoRA adapters
            lora_path = model_checkpoint.parent / "lora_adapters"
            if lora_path.exists():
                logger.info(f"Loading LoRA adapters from {lora_path}")
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(lora_path))
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Load test dataset
        if not test_data_path.exists():
            logger.warning(f"Test data not found: {test_data_path}. Skipping.")
            return None
        
        dataset = FiDDataset(
            data_path=test_data_path,
            tokenizer=tokenizer,
            max_input_length=512,
            max_output_length=200,
            max_passages=20,
        )
        
        if max_samples:
            dataset.examples = dataset.examples[:max_samples]
        
        dataloader = create_dataloader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=2,
        )
        
        # Evaluate
        all_predictions = []
        all_references = []
        token_counts = []
        
        logger.info(f"Running inference on {len(dataset)} samples...")
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Generate
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=200,
                    num_beams=4,
                )
                
                # Track token counts
                for seq in generated:
                    token_counts.append(seq.numel())
                
                # Decode
                pred_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
                ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                all_predictions.extend(pred_texts)
                all_references.extend(ref_texts)
        
        # Compute metrics
        logger.info(f"Computing metrics on {len(all_predictions)} predictions...")
        
        # ROUGE
        rouge_scores = evaluate_rouge(all_predictions, all_references)
        
        # BERTScore
        bertscore_dict = {}
        try:
            bertscore_dict = evaluate_bertscore(all_predictions, all_references)
        except Exception as e:
            logger.warning(f"Could not compute BERTScore: {e}")
        
        # EGF
        egf_values = []
        try:
            for pred, ref in zip(all_predictions, all_references):
                egf = compute_egf(pred, ref)
                if egf is not None:
                    egf_values.append(egf)
        except Exception as e:
            logger.warning(f"Could not compute EGF: {e}")
        
        # Compile results
        result = {
            "Model": model_name,
            "Dataset": dataset_name,
            "ROUGE-1": round(rouge_scores.get('rouge1', 0), 4),
            "ROUGE-2": round(rouge_scores.get('rouge2', 0), 4),
            "ROUGE-L": round(rouge_scores.get('rougeL', 0), 4),
            "BERTScore": round(bertscore_dict.get('bertscore_f1', 0), 4),
            "EGF_Metric": round(np.mean(egf_values), 4) if egf_values else 0,
            "Token_Count": int(np.mean(token_counts)) if token_counts else 0,
        }
        
        logger.info(f"Results: {result}")
        
        return result
    
    def generate_report(self, output_path: Path = Path("results/experiment_log.csv")):
        """Generate CSV report of all evaluations."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.results:
            logger.warning("No results to report.")
            return
        
        fieldnames = list(self.results[0].keys())
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        logger.info(f"✓ Saved evaluation report: {output_path}")
        
        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info("EVALUATION SUMMARY")
        logger.info(f"{'='*80}")
        
        for result in self.results:
            logger.info(
                f"{result['Model']:15} | {result['Dataset']:10} | "
                f"R1: {result['ROUGE-1']:.4f} | R2: {result['ROUGE-2']:.4f} | "
                f"RL: {result['ROUGE-L']:.4f} | Tokens: {result['Token_Count']:3d}"
            )


def main():
    """Main evaluation pipeline."""
    from ralfs.core.constants import PROCESSED_DIR
    
    runner = EvaluationRunner()
    
    # Define models and datasets to evaluate
    # Format: (model_name, dataset, test_data_path, checkpoint_dir)
    evaluations = [
        ("RALFS", "Arxiv", PROCESSED_DIR / "arxiv_test_chunks.jsonl", "checkpoints/ralfs_arxiv"),
        ("RALFS", "GovReport", PROCESSED_DIR / "govreport_test_chunks.jsonl", "checkpoints/ralfs_govreport"),
        # Add baseline evaluations if checkpoints exist
        # ("Standard FiD", "Arxiv", PROCESSED_DIR / "arxiv_test_chunks.jsonl", "checkpoints/fid_arxiv"),
        # etc.
    ]
    
    for model_name, dataset, test_path, checkpoint_dir in evaluations:
        checkpoint_path = Path(checkpoint_dir) / "best"
        tokenizer_path = checkpoint_path / "tokenizer"
        
        if checkpoint_path.exists() and tokenizer_path.exists():
            result = runner.evaluate_model(
                model_name=model_name,
                dataset_name=dataset,
                test_data_path=test_path,
                model_checkpoint=checkpoint_path,
                tokenizer_checkpoint=tokenizer_path,
                max_samples=100,  # Limit for speed
            )
            
            if result:
                runner.results.append(result)
        else:
            logger.warning(
                f"Checkpoint not found for {model_name}/{dataset}. "
                f"Expected: {checkpoint_path}"
            )
    
    # Generate report
    runner.generate_report()


if __name__ == "__main__":
    main()
