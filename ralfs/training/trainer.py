# ============================================================================
# File: ralfs/training/trainer.py
# ============================================================================
"""Enhanced training loop with LoRA, validation, metrics tracking, and W&B."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from ralfs.core.logging import get_logger
from ralfs.evaluation.metrics import evaluate_rouge
from ralfs.training.dataset import FiDDataset, create_dataloader

logger = get_logger(__name__)


class RALFSTrainer:
    """Enhanced trainer for FiD model with LoRA, validation, and W&B tracking."""

    def __init__(self, cfg):
        """Initialize trainer with configuration."""
        self.cfg = cfg

        # Training config
        self.train_config = cfg.train
        self.model_config = getattr(cfg.generator, "model", cfg.generator)
        self.lora_config = getattr(cfg.generator, "lora", None)

        # Get mixed precision setting
        training_cfg = getattr(self.train_config, "training", self.train_config)
        mixed_precision = getattr(training_cfg, "mixed_precision", "fp16")

        # Setup accelerator
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=getattr(training_cfg, "gradient_accumulation_steps", 1),
        )

        # Output directory
        output_dir = getattr(training_cfg, "output_dir", "checkpoints/default")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # W&B setup
        self.use_wandb = False
        wandb_cfg = getattr(self.train_config, "wandb", None)
        if wandb_cfg and getattr(wandb_cfg, "enabled", False):
            try:
                import wandb

                self.wandb = wandb
                self.use_wandb = True

                # Initialize W&B
                project = getattr(wandb_cfg, "project", "ralfs-experiments")
                entity = getattr(wandb_cfg, "entity", None)
                name = getattr(wandb_cfg, "name", None)
                tags = getattr(wandb_cfg, "tags", [])

                self.wandb.init(
                    project=project,
                    entity=entity,
                    name=name,
                    tags=tags,
                    config=self.cfg.to_dict(),
                )
                logger.info(f"✓ W&B initialized: {project}")
            except ImportError:
                logger.warning("wandb not installed. Install with: pip install wandb")
                self.use_wandb = False

        logger.info(f"Trainer initialized. Saving to {self.output_dir}")
        logger.info(f"Device: {self.accelerator.device}")
        logger.info(f"Mixed precision: {mixed_precision}")
        logger.info(f"W&B tracking: {'enabled' if self.use_wandb else 'disabled'}")

        # Model and tokenizer (initialized in setup)
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.global_step = 0
        self.best_metric = float("-inf")
        self.patience_counter = 0

    def setup_model(self):
        """Load model and apply LoRA."""
        model_name = getattr(self.model_config, "name", "google/flan-t5-large")

        logger.info(f"Loading model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Apply LoRA
        if self.lora_config and getattr(self.lora_config, "enabled", True):
            logger.info("Applying LoRA...")

            lora_r = getattr(self.lora_config, "r", 16)
            lora_alpha = getattr(self.lora_config, "alpha", 32)
            lora_dropout = getattr(self.lora_config, "dropout", 0.1)
            target_modules = getattr(self.lora_config, "target_modules", ["q", "v"])

            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
            )

            self.model = get_peft_model(self.model, peft_config)

            # Log trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"Trainable params: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)"
            )

        logger.info("Model loaded successfully")

    def setup_optimizer(self, train_dataloader: DataLoader):
        """Setup optimizer and scheduler."""
        # Get training config
        training_cfg = getattr(self.train_config, "training", self.train_config)
        model_cfg = getattr(self.train_config, "model", None)

        learning_rate = (
            getattr(model_cfg, "learning_rate", 5e-5)
            if model_cfg
            else getattr(training_cfg, "learning_rate", 5e-5)
        )
        weight_decay = (
            getattr(model_cfg, "weight_decay", 0.01)
            if model_cfg
            else getattr(training_cfg, "weight_decay", 0.01)
        )
        warmup_steps = getattr(training_cfg, "warmup_steps", 100)
        num_epochs = getattr(training_cfg, "num_epochs", 3)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        num_training_steps = len(train_dataloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        logger.info(
            f"Optimizer: AdamW (lr={learning_rate}, "
            f"warmup={warmup_steps}, steps={num_training_steps})"
        )

    def train(
        self,
        train_dataset: FiDDataset,
        eval_dataset: FiDDataset | None = None,
    ) -> dict[str, Any]:
        """
        Train the model with validation and metrics tracking.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional validation dataset

        Returns:
            Training statistics
        """
        # Setup model
        self.setup_model()

        # Get training config
        training_cfg = getattr(self.train_config, "training", self.train_config)

        batch_size = getattr(training_cfg, "batch_size", 1)
        num_epochs = getattr(training_cfg, "num_epochs", 3)
        logging_steps = getattr(training_cfg, "logging_steps", 50)
        save_steps = getattr(training_cfg, "save_steps", 1000)
        eval_steps = getattr(training_cfg, "eval_steps", 500)
        gradient_accumulation_steps = getattr(training_cfg, "gradient_accumulation_steps", 16)

        # Create dataloaders
        dataloader_cfg = getattr(self.train_config, "dataloader", None)
        num_workers = getattr(dataloader_cfg, "num_workers", 4) if dataloader_cfg else 4

        train_dataloader = create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        if eval_dataset:
            eval_dataloader = create_dataloader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        # Setup optimizer
        self.setup_optimizer(train_dataloader)

        # Prepare for distributed training
        self.model, self.optimizer, train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader
        )

        if eval_dataset:
            eval_dataloader = self.accelerator.prepare(eval_dataloader)

        # Training state
        self.global_step = 0
        self.best_metric = float("-inf")
        training_stats = {
            "train_losses": [],
            "eval_losses": [],
            "eval_rouge": [],
            "learning_rates": [],
            "global_steps": [],
            "epochs": [],
        }

        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"  Total steps: {len(train_dataloader) * num_epochs}")
        logger.info(f"  Batch size per device: {batch_size}")
        logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(
            f"  Effective batch size: {batch_size * gradient_accumulation_steps * self.accelerator.num_processes}"
        )

        start_time = time.time()

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0

            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                disable=not self.accelerator.is_main_process,
            )

            for step, batch in enumerate(progress_bar):
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                loss = outputs.loss / gradient_accumulation_steps

                # Backward pass
                self.accelerator.backward(loss)

                # Update (every gradient_accumulation_steps)
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping for stability
                    max_grad_norm = getattr(training_cfg, "max_grad_norm", 1.0)
                    if max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                # Track
                epoch_loss += loss.item() * gradient_accumulation_steps

                # Log
                if (
                    self.global_step % logging_steps == 0
                    and (step + 1) % gradient_accumulation_steps == 0
                ):
                    avg_loss = epoch_loss / (step + 1)
                    lr = self.scheduler.get_last_lr()[0]

                    progress_bar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{lr:.2e}",
                        }
                    )

                    training_stats["train_losses"].append(avg_loss)
                    training_stats["learning_rates"].append(lr)
                    training_stats["global_steps"].append(self.global_step)
                    training_stats["epochs"].append(epoch)

                    # W&B logging
                    if self.use_wandb:
                        self.wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/learning_rate": lr,
                                "train/epoch": epoch,
                                "train/step": self.global_step,
                                "train/perplexity": (
                                    np.exp(avg_loss) if avg_loss < 100 else float("inf")
                                ),
                            }
                        )

                # Save checkpoint
                if (
                    self.global_step % save_steps == 0
                    and (step + 1) % gradient_accumulation_steps == 0
                ):
                    self.save_checkpoint(f"step_{self.global_step}")

                # Evaluate
                if (
                    eval_dataset
                    and self.global_step % eval_steps == 0
                    and (step + 1) % gradient_accumulation_steps == 0
                ):
                    eval_metrics = self.evaluate(eval_dataloader)
                    training_stats["eval_losses"].append(eval_metrics["loss"])
                    training_stats["eval_rouge"].append(eval_metrics.get("rouge", {}))

                    # Check if best model
                    metric_for_best = getattr(training_cfg, "metric_for_best_model", "rougeL")
                    current_metric = eval_metrics.get("rouge", {}).get(
                        metric_for_best, eval_metrics["loss"]
                    )

                    # For loss, lower is better; for ROUGE, higher is better
                    if metric_for_best == "loss":
                        is_best = (
                            current_metric < self.best_metric
                            if self.best_metric != float("-inf")
                            else True
                        )
                    else:
                        is_best = current_metric > self.best_metric

                    if is_best:
                        self.best_metric = current_metric
                        self.save_checkpoint("best")
                        logger.info(f"✓ New best model! {metric_for_best}={current_metric:.4f}")
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1

                    # Early stopping
                    early_stopping_patience = getattr(training_cfg, "early_stopping_patience", None)
                    if early_stopping_patience and self.patience_counter >= early_stopping_patience:
                        logger.info(
                            f"Early stopping triggered after {self.patience_counter} evaluations without improvement"
                        )
                        break

            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_dataloader)

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} complete | "
                f"Avg Loss: {avg_epoch_loss:.4f} | "
                f"Time: {time.time() - start_time:.2f}s"
            )

            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch+1}")

            # Check early stopping at epoch level
            if hasattr(self, "patience_counter") and self.patience_counter >= getattr(
                training_cfg, "early_stopping_patience", float("inf")
            ):
                break

        # Save final model
        self.save_checkpoint("final")

        total_time = time.time() - start_time
        logger.info(f"Training complete! Total time: {total_time:.2f}s")

        # Save training stats
        stats_path = self.output_dir / "training_stats.json"
        with open(stats_path, "w") as f:
            json.dump(training_stats, f, indent=2)

        if self.use_wandb:
            self.wandb.finish()

        return training_stats

    def evaluate(self, eval_dataloader: DataLoader) -> dict[str, Any]:
        """
        Evaluate model on validation set with multiple metrics.

        Returns:
            Dict with loss, ROUGE, and other metrics
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_references = []

        logger.info("Running evaluation...")

        with torch.no_grad():
            for batch in tqdm(
                eval_dataloader, desc="Evaluating", disable=not self.accelerator.is_main_process
            ):
                # Compute loss
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                total_loss += outputs.loss.item()

                # Generate predictions (on subset for speed)
                if len(all_predictions) < 50:  # Limit to 50 samples
                    generated = self.model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=getattr(self.model_config, "max_output_length", 200),
                        num_beams=2,  # Faster beam search
                    )

                    # Decode
                    pred_texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                    ref_texts = self.tokenizer.batch_decode(
                        batch["labels"], skip_special_tokens=True
                    )

                    all_predictions.extend(pred_texts)
                    all_references.extend(ref_texts)

        avg_loss = total_loss / len(eval_dataloader)

        # Compute ROUGE on subset
        rouge_scores = {}
        if all_predictions:
            rouge_scores = evaluate_rouge(all_predictions, all_references)

        logger.info(f"Validation Loss: {avg_loss:.4f}")
        if rouge_scores:
            logger.info(f"Validation ROUGE-L: {rouge_scores.get('rougeL', 0):.4f}")

        # W&B logging
        if self.use_wandb:
            log_dict = {
                "eval/loss": avg_loss,
                "eval/step": self.global_step,
            }
            if rouge_scores:
                for key, val in rouge_scores.items():
                    log_dict[f"eval/{key}"] = val
            self.wandb.log(log_dict)

        self.model.train()

        return {
            "loss": avg_loss,
            "rouge": rouge_scores,
        }

    def save_checkpoint(self, name: str):
        """Save model checkpoint (both full state and LoRA adapters)."""
        if self.accelerator.is_main_process:
            save_path = self.output_dir / name
            save_path.mkdir(parents=True, exist_ok=True)

            # Save using accelerator (full state)
            self.accelerator.save_state(str(save_path / "accelerator"))

            # Save LoRA adapters separately (lighter)
            if isinstance(self.model, PeftModel) or hasattr(self.model, "save_pretrained"):
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                if hasattr(unwrapped_model, "save_pretrained"):
                    unwrapped_model.save_pretrained(str(save_path / "lora_adapters"))
                    logger.info(f"Saved LoRA adapters: {save_path / 'lora_adapters'}")

            # Save tokenizer
            self.tokenizer.save_pretrained(str(save_path / "tokenizer"))

            # Save config
            config_path = save_path / "config.json"
            with open(config_path, "w") as f:
                json.dump(self.cfg.to_dict(), f, indent=2)

            logger.info(f"✓ Checkpoint saved: {save_path}")


def train_model(cfg) -> dict[str, Any]:
    """
    Main training function (backward compatible).

    Args:
        cfg: Configuration object

    Returns:
        Training statistics
    """
    from ralfs.core.constants import PROCESSED_DIR

    trainer = RALFSTrainer(cfg)

    # Setup model first to get tokenizer
    trainer.setup_model()

    # Load datasets
    dataset = cfg.data.dataset
    train_path = PROCESSED_DIR / f"{dataset}_train_chunks.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}. " "Run 'ralfs preprocess' first."
        )

    train_dataset = FiDDataset(
        data_path=train_path,
        tokenizer=trainer.tokenizer,
        max_input_length=getattr(trainer.model_config, "max_input_length", 512),
        max_output_length=getattr(trainer.model_config, "max_output_length", 200),
        max_passages=getattr(cfg.generator, "adaptive_k_max", 20),
    )

    # Try to load validation dataset
    val_path = PROCESSED_DIR / f"{dataset}_validation_chunks.jsonl"
    eval_dataset = None

    if val_path.exists():
        logger.info(f"Loading validation data from {val_path}")
        eval_dataset = FiDDataset(
            data_path=val_path,
            tokenizer=trainer.tokenizer,
            max_input_length=getattr(trainer.model_config, "max_input_length", 512),
            max_output_length=getattr(trainer.model_config, "max_output_length", 200),
            max_passages=getattr(cfg.generator, "adaptive_k_max", 20),
        )
    else:
        logger.warning(f"Validation data not found: {val_path}. Training without validation.")

    # Train
    stats = trainer.train(train_dataset, eval_dataset)

    return stats
