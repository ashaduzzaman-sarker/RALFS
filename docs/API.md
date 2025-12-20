# RALFS API Documentation

## Core Modules

### Training

```python
from ralfs.training import RALFSTrainer, FiDDataset

# Create trainer
trainer = RALFSTrainer(config)

# Setup model
trainer.setup_model()

# Train
stats = trainer.train(train_dataset, eval_dataset)
```

### Evaluation

```python
from ralfs.evaluation import RALFSEvaluator, compare_systems

# Create evaluator
evaluator = RALFSEvaluator(metrics=['rouge', 'bertscore', 'egf'])

# Evaluate
individual, aggregated = evaluator.evaluate(predictions, references)

# Compare systems
comparison = compare_systems(system1_results, system2_results)
```

### Reproducibility

```python
from ralfs.utils import set_seed, ExperimentTracker

# Set seed
set_seed(42, deterministic=True)

# Track experiment
tracker = ExperimentTracker("experiments/exp1", seed=42)
tracker.log_config(config_dict)
tracker.log_metric("rouge_l", 0.45)
tracker.save()
```

## Key Classes

### RALFSTrainer

Main training class with LoRA, mixed precision, and W&B integration.

**Methods:**
- `setup_model()`: Load model and apply LoRA
- `setup_optimizer(dataloader)`: Create optimizer and scheduler
- `train(train_dataset, eval_dataset)`: Main training loop
- `evaluate(dataloader)`: Evaluation with metrics
- `save_checkpoint(name)`: Save model checkpoint

### RALFSEvaluator

Comprehensive evaluator with statistical tests.

**Methods:**
- `evaluate(predictions, references)`: Evaluate with all metrics
- `_bootstrap_ci(scores)`: Compute bootstrap confidence intervals
- `save_results(results, output_dir)`: Save in multiple formats

### ExperimentTracker

Track experiments for reproducibility.

**Methods:**
- `log_config(config)`: Log configuration
- `log_metric(name, value)`: Log single metric
- `log_metrics(metrics)`: Log multiple metrics
- `save(notes)`: Save experiment metadata

## Functions

### compare_systems

```python
def compare_systems(
    system1_results: List[EvaluationResult],
    system2_results: List[EvaluationResult],
    alpha: float = 0.05,
) -> Dict[str, Any]
```

Compare two systems with paired t-test and bootstrap testing.

**Returns:**
- Difference statistics
- P-values (t-test and bootstrap)
- Cohen's d effect size
- Significance flags

### set_seed

```python
def set_seed(seed: int = 42, deterministic: bool = True) -> None
```

Set all random seeds for reproducibility.

### compute_egf

```python
def compute_egf(
    reference: str,
    generated: str,
    return_details: bool = False,
) -> float | Dict[str, Any]
```

Compute Entity Grid Faithfulness score.
