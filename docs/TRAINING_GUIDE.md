# RALFS Training Guide

Complete guide for training RALFS models for conference paper submissions.

## Quick Start

### 1. Environment Setup
```bash
pip install -e .
python -m spacy download en_core_web_sm
```

### 2. Data Preprocessing
```bash
ralfs preprocess --dataset arxiv --max-samples 1000
ralfs build-index --dataset arxiv
```

### 3. Training
```bash
ralfs train --dataset arxiv --config configs/train/default.yaml
```

## Reproducibility

### Setting Seeds
```python
from ralfs.utils import set_seed
set_seed(42, deterministic=True)
```

### Experiment Tracking
```python
from ralfs.utils import ExperimentTracker

tracker = ExperimentTracker("experiments/exp1", seed=42)
tracker.log_config({"lr": 5e-5, "batch_size": 16})
tracker.log_metric("val_rouge_l", 0.45)
tracker.save(notes="Baseline experiment")
```

## Evaluation

### Statistical Significance Testing
```python
from ralfs.evaluation import compare_systems

comparison = compare_systems(baseline_results, ralfs_results)
print(f"ROUGE-L: Δ={comparison['rougeL_diff_mean']:.4f}, p={comparison['rougeL_p_value']:.4f}")
```

## Best Practices for Conference Papers

1. Report all hyperparameters
2. Use multiple seeds (3-5 runs)
3. Report confidence intervals (bootstrap)
4. Statistical significance testing
5. Share code and configurations
6. Report system details (GPU, time, memory)
7. Conduct ablation studies

## References

- Hu et al. (2021). LoRA: Low-Rank Adaptation. ICLR.
- Lin (2004). ROUGE: Automatic Evaluation of Summaries. ACL.
- Zhang et al. (2020). BERTScore. ICLR.
- Barzilay & Lapata (2008). Entity-based coherence. CL.
