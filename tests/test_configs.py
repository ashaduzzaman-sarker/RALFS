"""Tests for ralfs.core.config module."""

import pytest
from pathlib import Path
from omegaconf import OmegaConf
from ralfs.core.config import (
    DataConfig,
    RetrieverConfig,
    GeneratorConfig,
    TrainConfig,
    RALFSConfig,
    load_config,
)


class TestDataConfig:
    """Tests for DataConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()
        assert config.dataset == "arxiv"
        assert config.split == "train"
        assert config.chunk_size == 512
        assert config.overlap == 128
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = DataConfig(
            dataset="pubmed",
            split="test",
            chunk_size=1024,
            max_samples=100,
        )
        assert config.dataset == "pubmed"
        assert config.split == "test"
        assert config.chunk_size == 1024
        assert config.max_samples == 100


class TestRetrieverConfig:
    """Tests for RetrieverConfig."""
    
    def test_default_retriever_config(self):
        """Test default retriever configuration."""
        config = RetrieverConfig()
        assert config.type == "hybrid"
        assert config.k_final == 20
        assert config.use_reranker is True
    
    def test_retriever_k_values(self):
        """Test retrieval k values."""
        config = RetrieverConfig(
            k_dense=50,
            k_sparse=50,
            k_colbert=50,
            k_final=10,
        )
        assert config.k_dense == 50
        assert config.k_sparse == 50
        assert config.k_colbert == 50
        assert config.k_final == 10


class TestGeneratorConfig:
    """Tests for GeneratorConfig."""
    
    def test_default_generator_config(self):
        """Test default generator configuration."""
        config = GeneratorConfig()
        assert config.model_name == "google/flan-t5-large"
        assert config.max_input_length == 8192
        assert config.adaptive_k is True
        assert config.use_lora is True
    
    def test_lora_parameters(self):
        """Test LoRA parameters."""
        config = GeneratorConfig(
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.2,
        )
        assert config.lora_r == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == 0.2


class TestTrainConfig:
    """Tests for TrainConfig."""
    
    def test_default_train_config(self):
        """Test default training configuration."""
        config = TrainConfig()
        assert config.num_epochs == 3
        assert config.batch_size == 4
        assert config.learning_rate == 1e-4
        assert config.fp16 is True
    
    def test_wandb_config(self):
        """Test W&B configuration."""
        config = TrainConfig(
            use_wandb=True,
            wandb_project="ralfs-test",
        )
        assert config.use_wandb is True
        assert config.wandb_project == "ralfs-test"


class TestRALFSConfig:
    """Tests for main RALFSConfig."""
    
    def test_default_config(self):
        """Test default RALFS configuration."""
        config = RALFSConfig()
        assert config.task == "train"
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.retriever, RetrieverConfig)
        assert isinstance(config.generator, GeneratorConfig)
        assert isinstance(config.train, TrainConfig)
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = RALFSConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "task" in config_dict
        assert "data" in config_dict
        assert "retriever" in config_dict
    
    def test_save_and_load_config(self, tmp_path):
        """Test saving and loading config."""
        config = RALFSConfig(task="generate")
        config.data.dataset = "pubmed"
        config.retriever.k_final = 15
        
        config_path = tmp_path / "test_config.yaml"
        config.save(config_path)
        
        assert config_path.exists()
        loaded_config = RALFSConfig.from_yaml(config_path)
        
        assert loaded_config.task == "generate"
        assert loaded_config.data.dataset == "pubmed"
        assert loaded_config.retriever.k_final == 15


class TestLoadConfig:
    """Tests for load_config function."""
    
    def test_load_config_with_overrides(self, tmp_path):
        """Test loading config with overrides."""
        # Create a basic config file
        config_path = tmp_path / "test.yaml"
        config = RALFSConfig()
        config.save(config_path)
        
        # Load with overrides
        loaded_config = load_config(
            config_path,
            overrides=[
                "data.dataset=govreport",
                "train.batch_size=8",
                "retriever.k_final=25",
            ]
        )
        
        assert loaded_config.data.dataset == "govreport"
        assert loaded_config.train.batch_size == 8
        assert loaded_config.retriever.k_final == 25
    
    def test_load_nonexistent_config_uses_defaults(self, tmp_path):
        """Test that loading nonexistent config uses defaults."""
        config_path = tmp_path / "nonexistent.yaml"
        config = load_config(config_path)
        
        # Should use default values
        assert config.task == "train"
        assert config.data.dataset == "arxiv"
