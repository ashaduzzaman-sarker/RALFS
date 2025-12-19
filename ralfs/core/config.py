# ============================================================================
# File: ralfs/core/config.py
# ============================================================================
"""Configuration management using Hydra and OmegaConf."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Dict, List
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import logging

from .constants import (
    ROOT_DIR,
    CONFIGS_DIR,
    DEFAULT_DENSE_MODEL,
    DEFAULT_COLBERT_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_K_FINAL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
)

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data processing configuration."""
    dataset: str = "arxiv"
    split: str = "train"
    max_samples: Optional[int] = None
    chunk_size: int = 512
    overlap: int = 128
    min_chunk_size: int = 100
    chunking_strategy: str = "semantic"
    num_workers: int = 4
    batch_size: int = 32
    seed: int = 42
    use_cache: bool = True
    cache_dir: str = ".cache/ralfs"


@dataclass
class RetrieverConfig:
    """Retrieval configuration."""
    type: str = "hybrid"  # dense, sparse, colbert, hybrid
    dense_model: str = DEFAULT_DENSE_MODEL
    colbert_model: str = DEFAULT_COLBERT_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    k_dense: int = 100
    k_sparse: int = 100
    k_colbert: int = 100
    k_rerank: int = 50
    k_final: int = DEFAULT_K_FINAL
    rrf_k: int = 60
    use_reranker: bool = True


@dataclass
class GeneratorConfig:
    """Generator configuration."""
    model_name: str = DEFAULT_GENERATOR_MODEL
    max_input_length: int = 8192
    max_output_length: int = 512
    num_beams: int = 4
    length_penalty: float = 1.0
    adaptive_k: bool = True
    adaptive_k_min: int = 5
    adaptive_k_max: int = 30
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1


@dataclass
class TrainConfig:
    """Training configuration."""
    output_dir: str = "checkpoints/default"
    num_epochs: int = DEFAULT_NUM_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    fp16: bool = True
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    seed: int = 42


def _to_dict_safe(cfg: Any) -> Dict[str, Any]:
    """
    Safely convert config to dict, handling both DictConfig and plain dict.
    
    Args:
        cfg: Configuration object (DictConfig, dict, or None)
    
    Returns:
        Plain Python dict
    """
    if cfg is None:
        return {}
    elif isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    elif isinstance(cfg, dict):
        return cfg
    else:
        # Try to convert to dict
        try:
            return dict(cfg)
        except:
            return {}


@dataclass
class RALFSConfig:
    """Main RALFS configuration."""
    task: str = "train"  # preprocess, train, generate, evaluate
    data: DataConfig = field(default_factory=DataConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "RALFSConfig":
        """
        Create RALFSConfig from Hydra DictConfig.
        
        Args:
            cfg: Hydra configuration object
        
        Returns:
            RALFSConfig instance
        """
        # Extract task
        task = cfg.get("task", "train")
        
        # Safely extract and convert each section
        data_dict = _to_dict_safe(cfg.get("data", {}))
        retriever_dict = _to_dict_safe(cfg.get("retriever", {}))
        generator_dict = _to_dict_safe(cfg.get("generator", {}))
        train_dict = _to_dict_safe(cfg.get("train", {}))
        
        # Handle nested configs in retriever (dense, sparse, colbert, etc.)
        # Flatten nested structures if they exist
        if "dense" in retriever_dict and isinstance(retriever_dict["dense"], dict):
            # Extract dense.model to dense_model
            if "model" in retriever_dict["dense"]:
                retriever_dict["dense_model"] = retriever_dict["dense"]["model"]
            if "k" in retriever_dict["dense"]:
                retriever_dict["k_dense"] = retriever_dict["dense"]["k"]
        
        if "sparse" in retriever_dict and isinstance(retriever_dict["sparse"], dict):
            if "k" in retriever_dict["sparse"]:
                retriever_dict["k_sparse"] = retriever_dict["sparse"]["k"]
        
        if "colbert" in retriever_dict and isinstance(retriever_dict["colbert"], dict):
            if "model" in retriever_dict["colbert"]:
                retriever_dict["colbert_model"] = retriever_dict["colbert"]["model"]
            if "k" in retriever_dict["colbert"]:
                retriever_dict["k_colbert"] = retriever_dict["colbert"]["k"]
        
        if "reranker" in retriever_dict and isinstance(retriever_dict["reranker"], dict):
            if "model" in retriever_dict["reranker"]:
                retriever_dict["reranker_model"] = retriever_dict["reranker"]["model"]
            if "enabled" in retriever_dict["reranker"]:
                retriever_dict["use_reranker"] = retriever_dict["reranker"]["enabled"]
            if "top_k" in retriever_dict["reranker"]:
                retriever_dict["k_final"] = retriever_dict["reranker"]["top_k"]
        
        if "fusion" in retriever_dict and isinstance(retriever_dict["fusion"], dict):
            if "rrf_k" in retriever_dict["fusion"]:
                retriever_dict["rrf_k"] = retriever_dict["fusion"]["rrf_k"]
        
        # Handle nested configs in generator (model, adaptive, lora)
        if "model" in generator_dict and isinstance(generator_dict["model"], dict):
            model_cfg = generator_dict["model"]
            if "name" in model_cfg:
                generator_dict["model_name"] = model_cfg["name"]
            if "max_input_length" in model_cfg:
                generator_dict["max_input_length"] = model_cfg["max_input_length"]
            if "max_output_length" in model_cfg:
                generator_dict["max_output_length"] = model_cfg["max_output_length"]
            if "num_beams" in model_cfg:
                generator_dict["num_beams"] = model_cfg["num_beams"]
            if "length_penalty" in model_cfg:
                generator_dict["length_penalty"] = model_cfg["length_penalty"]
        
        if "adaptive" in generator_dict and isinstance(generator_dict["adaptive"], dict):
            adaptive_cfg = generator_dict["adaptive"]
            if "enabled" in adaptive_cfg:
                generator_dict["adaptive_k"] = adaptive_cfg["enabled"]
            if "min_k" in adaptive_cfg:
                generator_dict["adaptive_k_min"] = adaptive_cfg["min_k"]
            if "max_k" in adaptive_cfg:
                generator_dict["adaptive_k_max"] = adaptive_cfg["max_k"]
        
        if "lora" in generator_dict and isinstance(generator_dict["lora"], dict):
            lora_cfg = generator_dict["lora"]
            if "enabled" in lora_cfg:
                generator_dict["use_lora"] = lora_cfg["enabled"]
            if "r" in lora_cfg:
                generator_dict["lora_r"] = lora_cfg["r"]
            if "alpha" in lora_cfg:
                generator_dict["lora_alpha"] = lora_cfg["alpha"]
            if "dropout" in lora_cfg:
                generator_dict["lora_dropout"] = lora_cfg["dropout"]
        
        # Handle nested configs in train (model, training, dataloader, wandb)
        if "model" in train_dict and isinstance(train_dict["model"], dict):
            model_cfg = train_dict["model"]
            if "learning_rate" in model_cfg:
                train_dict["learning_rate"] = model_cfg["learning_rate"]
            if "weight_decay" in model_cfg:
                train_dict["weight_decay"] = model_cfg["weight_decay"]
        
        if "training" in train_dict and isinstance(train_dict["training"], dict):
            training_cfg = train_dict["training"]
            # Merge training config into train_dict
            for key, value in training_cfg.items():
                if key not in train_dict:
                    train_dict[key] = value
        
        if "wandb" in train_dict and isinstance(train_dict["wandb"], dict):
            wandb_cfg = train_dict["wandb"]
            if "enabled" in wandb_cfg:
                train_dict["use_wandb"] = wandb_cfg["enabled"]
            if "project" in wandb_cfg:
                train_dict["wandb_project"] = wandb_cfg["project"]
        
        # Filter to only include fields defined in dataclasses
        data_fields = {f.name for f in DataConfig.__dataclass_fields__.values()}
        retriever_fields = {f.name for f in RetrieverConfig.__dataclass_fields__.values()}
        generator_fields = {f.name for f in GeneratorConfig.__dataclass_fields__.values()}
        train_fields = {f.name for f in TrainConfig.__dataclass_fields__.values()}
        
        data_dict = {k: v for k, v in data_dict.items() if k in data_fields}
        retriever_dict = {k: v for k, v in retriever_dict.items() if k in retriever_fields}
        generator_dict = {k: v for k, v in generator_dict.items() if k in generator_fields}
        train_dict = {k: v for k, v in train_dict.items() if k in train_fields}
        
        # Create config objects
        return cls(
            task=task,
            data=DataConfig(**data_dict),
            retriever=RetrieverConfig(**retriever_dict),
            generator=GeneratorConfig(**generator_dict),
            train=TrainConfig(**train_dict),
        )
    
    @classmethod
    def from_yaml(cls, config_path: Path | str) -> "RALFSConfig":
        """Load config from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        cfg = OmegaConf.load(config_path)
        return cls.from_hydra(cfg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task": self.task,
            "data": asdict(self.data),
            "retriever": asdict(self.retriever),
            "generator": asdict(self.generator),
            "train": asdict(self.train),
        }
    
    def save(self, path: Path | str) -> None:
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and then to OmegaConf for better YAML formatting
        config_dict = self.to_dict()
        cfg = OmegaConf.create(config_dict)
        OmegaConf.save(cfg, path)
        logger.info(f"Saved config to {path}")


def load_config(
    config_path: Optional[Path | str] = None,
    overrides: Optional[List[str]] = None,
) -> RALFSConfig:
    """
    Load RALFS configuration with optional overrides.
    
    Args:
        config_path: Path to config YAML (default: configs/ralfs.yaml)
        overrides: List of config overrides in dotted notation
                   e.g., ["data.dataset=pubmed", "train.batch_size=8"]
    
    Returns:
        RALFSConfig instance
    
    Example:
        >>> config = load_config("configs/ralfs.yaml", ["data.dataset=pubmed"])
    """
    if config_path is None:
        config_path = CONFIGS_DIR / "ralfs.yaml"
    
    config_path = Path(config_path)
    
    # Load base config
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
    else:
        logger.warning(f"Config not found: {config_path}, using defaults")
        cfg = OmegaConf.create({"task": "train"})
    
    # Apply overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    
    return RALFSConfig.from_hydra(cfg)