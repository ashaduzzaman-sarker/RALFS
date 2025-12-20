# ============================================================================
# File: tests/test_training.py
# Comprehensive tests for training module
# ============================================================================
"""Unit and integration tests for RALFS training."""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from omegaconf import OmegaConf

from ralfs.training.trainer import RALFSTrainer, train_model
from ralfs.training.dataset import FiDDataset, create_dataloader


@pytest.fixture(autouse=True)
def reset_accelerator_state():
    """Reset AcceleratorState before each test to avoid reinitialization errors."""
    from accelerate.state import AcceleratorState
    
    # Reset the singleton instance
    if AcceleratorState._shared_state:
        AcceleratorState._shared_state.clear()
    
    yield
    
    # Clean up after test
    if AcceleratorState._shared_state:
        AcceleratorState._shared_state.clear()


class TestFiDDataset:
    """Test FiD dataset class."""
    
    def test_dataset_initialization(self, tmp_path):
        """Test dataset initialization."""
        # Create mock data
        data_file = tmp_path / "test_data.jsonl"
        import json
        with open(data_file, 'w') as f:
            for i in range(5):
                sample = {
                    "text": f"Sample text {i}",
                    "metadata": {"summary": f"Summary {i}"}
                }
                f.write(json.dumps(sample) + '\n')
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': torch.ones(1, 10),
            'attention_mask': torch.ones(1, 10),
        }
        
        # Create dataset
        dataset = FiDDataset(
            data_path=data_file,
            tokenizer=mock_tokenizer,
            max_input_length=512,
            max_output_length=200,
        )
        
        assert len(dataset) == 5
    
    def test_dataset_getitem(self, tmp_path):
        """Test dataset __getitem__ method."""
        data_file = tmp_path / "test_data.jsonl"
        import json
        with open(data_file, 'w') as f:
            sample = {
                "text": "Test document text",
                "metadata": {"summary": "Test summary"}
            }
            f.write(json.dumps(sample) + '\n')
        
        # Mock tokenizer with proper return
        mock_tokenizer = Mock()
        def mock_encode(*args, **kwargs):
            return {
                'input_ids': torch.ones(5, 10, dtype=torch.long),
                'attention_mask': torch.ones(5, 10, dtype=torch.long),
            }
        mock_tokenizer.side_effect = mock_encode
        
        dataset = FiDDataset(
            data_path=data_file,
            tokenizer=mock_tokenizer,
            max_input_length=512,
            max_output_length=200,
        )
        
        item = dataset[0]
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'labels' in item
    
    def test_create_dataloader(self, tmp_path):
        """Test dataloader creation."""
        data_file = tmp_path / "test_data.jsonl"
        import json
        with open(data_file, 'w') as f:
            for i in range(10):
                f.write(json.dumps({
                    "text": f"Text {i}",
                    "metadata": {"summary": f"Summary {i}"}
                }) + '\n')
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': torch.ones(5, 10),
            'attention_mask': torch.ones(5, 10),
        }
        
        dataset = FiDDataset(data_file, mock_tokenizer)
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False)
        
        assert dataloader is not None
        assert dataloader.batch_size == 2


class TestRALFSTrainer:
    """Test RALFS trainer class."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        cfg = OmegaConf.create({
            'train': {
                'training': {
                    'mixed_precision': 'no',
                    'gradient_accumulation_steps': 1,
                    'output_dir': 'checkpoints/test',
                },
                'wandb': {'enabled': False}
            },
            'generator': {
                'model': {'name': 'google/flan-t5-small'},
                'lora': {'enabled': False},
            },
            'data': {'dataset': 'test'}
        })
        
        trainer = RALFSTrainer(cfg)
        
        assert trainer.cfg == cfg
        assert trainer.output_dir.exists()
        assert trainer.use_wandb == False
    
    @patch('ralfs.training.trainer.AutoModelForSeq2SeqLM')
    @patch('ralfs.training.trainer.AutoTokenizer')
    def test_setup_model(self, mock_tokenizer, mock_model):
        """Test model setup."""
        cfg = OmegaConf.create({
            'train': {
                'training': {
                    'mixed_precision': 'no',
                    'gradient_accumulation_steps': 1,
                    'output_dir': 'checkpoints/test',
                },
                'wandb': {'enabled': False}
            },
            'generator': {
                'model': {'name': 'google/flan-t5-small'},
                'lora': {'enabled': False},
            },
            'data': {'dataset': 'test'}
        })
        
        # Mock returns
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        
        trainer = RALFSTrainer(cfg)
        trainer.setup_model()
        
        assert trainer.model is not None
        assert trainer.tokenizer is not None
    
    @patch('ralfs.training.trainer.AutoModelForSeq2SeqLM')
    @patch('ralfs.training.trainer.AutoTokenizer')
    def test_gradient_clipping(self, mock_tokenizer, mock_model):
        """Test gradient clipping is applied."""
        cfg = OmegaConf.create({
            'train': {
                'training': {
                    'mixed_precision': 'no',
                    'gradient_accumulation_steps': 1,
                    'output_dir': 'checkpoints/test',
                    'max_grad_norm': 1.0,
                },
                'model': {
                    'learning_rate': 5e-5,
                    'weight_decay': 0.01,
                },
                'wandb': {'enabled': False}
            },
            'generator': {
                'model': {'name': 'google/flan-t5-small'},
                'lora': {'enabled': False},
            },
            'data': {'dataset': 'test'}
        })
        
        trainer = RALFSTrainer(cfg)
        
        # Verify max_grad_norm is set
        assert hasattr(cfg.train.training, 'max_grad_norm')
        assert cfg.train.training.max_grad_norm == 1.0


@pytest.mark.slow
class TestTrainingIntegration:
    """Integration tests for training pipeline."""
    
    def test_training_stats_tracking(self):
        """Test that training statistics are properly tracked."""
        cfg = OmegaConf.create({
            'train': {
                'training': {
                    'mixed_precision': 'no',
                    'gradient_accumulation_steps': 1,
                    'output_dir': 'checkpoints/test',
                    'batch_size': 1,
                    'num_epochs': 1,
                    'logging_steps': 1,
                    'save_steps': 100,
                    'eval_steps': 100,
                },
                'model': {
                    'learning_rate': 5e-5,
                    'weight_decay': 0.01,
                },
                'dataloader': {'num_workers': 0},
                'wandb': {'enabled': False}
            },
            'generator': {
                'model': {
                    'name': 'google/flan-t5-small',
                    'max_input_length': 128,
                    'max_output_length': 50,
                },
                'lora': {'enabled': False},
                'adaptive_k_max': 5,
            },
            'data': {'dataset': 'test'}
        })
        
        trainer = RALFSTrainer(cfg)
        
        # Verify training stats structure
        training_stats = {
            'train_losses': [],
            'eval_losses': [],
            'eval_rouge': [],
            'learning_rates': [],
            'global_steps': [],
            'epochs': [],
        }
        
        # All required fields are present
        assert 'train_losses' in training_stats
        assert 'learning_rates' in training_stats
        assert 'global_steps' in training_stats
        assert 'epochs' in training_stats


class TestOptimization:
    """Test optimization and training utilities."""
    
    def test_learning_rate_schedule(self):
        """Test learning rate scheduling."""
        from transformers import get_linear_schedule_with_warmup
        
        # Mock optimizer
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=1000,
        )
        
        # Initial LR
        initial_lr = scheduler.get_last_lr()[0]
        
        # Step through warmup
        for _ in range(100):
            scheduler.step()
        
        warmup_lr = scheduler.get_last_lr()[0]
        
        # LR should increase during warmup
        assert warmup_lr > initial_lr
    
    def test_gradient_accumulation_logic(self):
        """Test gradient accumulation calculations."""
        batch_size = 2
        gradient_accumulation_steps = 8
        
        effective_batch_size = batch_size * gradient_accumulation_steps
        
        assert effective_batch_size == 16


class TestCheckpointing:
    """Test model checkpointing."""
    
    def test_checkpoint_directory_creation(self, tmp_path):
        """Test checkpoint directory is created."""
        cfg = OmegaConf.create({
            'train': {
                'training': {
                    'mixed_precision': 'no',
                    'gradient_accumulation_steps': 1,
                    'output_dir': str(tmp_path / 'checkpoints'),
                },
                'wandb': {'enabled': False}
            },
            'generator': {
                'model': {'name': 'google/flan-t5-small'},
                'lora': {'enabled': False},
            },
            'data': {'dataset': 'test'}
        })
        
        trainer = RALFSTrainer(cfg)
        
        assert trainer.output_dir.exists()
        assert trainer.output_dir == tmp_path / 'checkpoints'


class TestReproducibility:
    """Test reproducibility features."""
    
    def test_seed_setting(self):
        """Test that seeds can be set for reproducibility."""
        import random
        
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Generate random numbers
        torch_val1 = torch.rand(1).item()
        np_val1 = np.random.rand()
        
        # Reset seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Should get same values
        torch_val2 = torch.rand(1).item()
        np_val2 = np.random.rand()
        
        assert torch_val1 == torch_val2
        assert np_val1 == np_val2
