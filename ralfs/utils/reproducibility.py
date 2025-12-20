# ============================================================================
# File: ralfs/utils/reproducibility.py
# Reproducibility utilities for research-grade experiments
# ============================================================================
"""
Utilities for ensuring reproducibility in experiments.

Essential for conference paper submissions requiring reproducible results.
Implements seed setting, deterministic operations, and experiment tracking.
"""

from __future__ import annotations
import os
import random
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch

from ralfs.core.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CUDA (deterministic algorithms if requested)
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic CUDA operations
                      (may reduce performance but ensures reproducibility)
    
    Example:
        >>> set_seed(42, deterministic=True)
        >>> # All random operations will now be reproducible
    
    Note:
        Deterministic mode may impact performance. For training with
        randomness (e.g., dropout), use deterministic=False for speed
        while still benefiting from seed reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    if deterministic:
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # PyTorch 1.8+: Use deterministic algorithms
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        
        logger.info(f"Seed set to {seed} with deterministic mode enabled")
    else:
        # Allow non-deterministic for better performance
        torch.backends.cudnn.benchmark = True
        logger.info(f"Seed set to {seed} without deterministic mode")
    
    # Set environment variables for additional libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"✓ Reproducibility configured (seed={seed}, deterministic={deterministic})")


def get_experiment_config() -> Dict[str, Any]:
    """
    Collect environment and system configuration for reproducibility.
    
    Returns:
        Dictionary with system configuration including:
        - PyTorch version
        - CUDA version
        - Python version
        - NumPy version
        - System info
    
    Example:
        >>> config = get_experiment_config()
        >>> print(config['pytorch_version'])
    """
    config = {
        'pytorch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        config['cuda_version'] = torch.version.cuda
        config['cudnn_version'] = torch.backends.cudnn.version()
        config['num_gpus'] = torch.cuda.device_count()
        config['gpu_names'] = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]
    
    # Python version
    import sys
    config['python_version'] = sys.version
    
    # Platform info
    import platform
    config['platform'] = {
        'system': platform.system(),
        'release': platform.release(),
        'machine': platform.machine(),
    }
    
    return config


def save_experiment_metadata(
    output_dir: Path | str,
    config: Dict[str, Any],
    metrics: Optional[Dict[str, float]] = None,
    notes: Optional[str] = None,
) -> Path:
    """
    Save experiment metadata for reproducibility.
    
    Args:
        output_dir: Directory to save metadata
        config: Configuration dictionary
        metrics: Optional metrics dictionary
        notes: Optional experiment notes
    
    Returns:
        Path to saved metadata file
    
    Example:
        >>> save_experiment_metadata(
        ...     "results/exp1",
        ...     config={'lr': 5e-5, 'batch_size': 8},
        ...     metrics={'rouge_l': 0.45},
        ...     notes="Baseline experiment with FiD"
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine all metadata
    metadata = {
        'config': config,
        'system': get_experiment_config(),
        'metrics': metrics or {},
        'notes': notes or "",
    }
    
    # Add timestamp
    from datetime import datetime
    metadata['timestamp'] = datetime.now().isoformat()
    
    # Save to JSON
    metadata_path = output_dir / 'experiment_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"✓ Experiment metadata saved: {metadata_path}")
    
    return metadata_path


class ExperimentTracker:
    """
    Track experiments for reproducibility and comparison.
    
    Features:
    - Automatic configuration saving
    - Metric logging
    - Seed management
    - System info recording
    
    Example:
        >>> tracker = ExperimentTracker("experiments/exp1", seed=42)
        >>> tracker.log_config({"lr": 5e-5, "batch_size": 8})
        >>> tracker.log_metric("rouge_l", 0.45)
        >>> tracker.save()
    """
    
    def __init__(
        self,
        output_dir: Path | str,
        seed: int = 42,
        deterministic: bool = True,
    ):
        """
        Initialize experiment tracker.
        
        Args:
            output_dir: Directory to save experiment data
            seed: Random seed
            deterministic: Use deterministic operations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed
        set_seed(seed, deterministic)
        
        # Initialize tracking data
        self.config = {}
        self.metrics = {}
        self.seed = seed
        self.deterministic = deterministic
        
        logger.info(f"Experiment tracker initialized: {self.output_dir}")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration parameters."""
        self.config.update(config)
        logger.info(f"Config logged: {len(config)} parameters")
    
    def log_metric(self, name: str, value: float) -> None:
        """Log a metric value."""
        self.metrics[name] = value
        logger.info(f"Metric logged: {name}={value:.4f}")
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log multiple metrics."""
        self.metrics.update(metrics)
        logger.info(f"Metrics logged: {len(metrics)} metrics")
    
    def save(self, notes: Optional[str] = None) -> Path:
        """
        Save all tracked experiment data.
        
        Args:
            notes: Optional notes about the experiment
        
        Returns:
            Path to saved metadata file
        """
        return save_experiment_metadata(
            self.output_dir,
            config={
                **self.config,
                'seed': self.seed,
                'deterministic': self.deterministic,
            },
            metrics=self.metrics,
            notes=notes,
        )
    
    def log_training_history(self, history: Dict[str, list]) -> None:
        """
        Save training history (losses, metrics over time).
        
        Args:
            history: Dictionary with lists of values over training
        """
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"✓ Training history saved: {history_path}")


def verify_reproducibility(
    func,
    *args,
    seed: int = 42,
    n_runs: int = 3,
    **kwargs
) -> bool:
    """
    Verify that a function produces reproducible results.
    
    Runs the function multiple times with the same seed and checks
    if outputs are identical.
    
    Args:
        func: Function to test
        *args: Arguments to pass to function
        seed: Random seed
        n_runs: Number of runs to verify
        **kwargs: Keyword arguments to pass to function
    
    Returns:
        True if all runs produce identical results
    
    Example:
        >>> def train_model():
        ...     return model.train()
        >>> is_reproducible = verify_reproducibility(train_model, seed=42)
    """
    results = []
    
    for i in range(n_runs):
        set_seed(seed, deterministic=True)
        result = func(*args, **kwargs)
        results.append(result)
        logger.info(f"Run {i+1}/{n_runs} completed")
    
    # Check if all results are identical
    # (Handles tensors, arrays, and basic types)
    first = results[0]
    
    for i, result in enumerate(results[1:], start=2):
        if isinstance(first, torch.Tensor):
            if not torch.allclose(first, result):
                logger.warning(f"Run {i} differs from run 1")
                return False
        elif isinstance(first, np.ndarray):
            if not np.allclose(first, result):
                logger.warning(f"Run {i} differs from run 1")
                return False
        else:
            if first != result:
                logger.warning(f"Run {i} differs from run 1")
                return False
    
    logger.info(f"✓ Function is reproducible across {n_runs} runs")
    return True
