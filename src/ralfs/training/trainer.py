# src/ralfs/training/trainer.py
from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from tqdm.auto import tqdm
from ralfs.core.logging import get_logger
from ralfs.training.dataset import FiDDataset

logger = get_logger(__name__)

def train(cfg):
    accelerator = Accelerator(mixed_precision=cfg.training.mixed_precision)

    # Output directory
    output_dir = Path("checkpoints") / f"{cfg.data.dataset}_fid"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving checkpoints to {output_dir}")

    # Model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.name)

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    dataset = FiDDataset(cfg, split="train")
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        num_workers=0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.model.lr)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    for epoch in range(cfg.model.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
            # Tokenize inputs
            encoded_inputs = tokenizer(
                batch["inputs"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(accelerator.device)

            # Tokenize labels
            encoded_labels = tokenizer(
                batch["summaries"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).input_ids.to(accelerator.device)

            loss = model(**encoded_inputs, labels=encoded_labels).loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{cfg.model.epochs} | Avg Loss: {avg_loss:.4f}")
        accelerator.save_state(str(output_dir / f"epoch_{epoch+1}"))

    accelerator.save_state(str(output_dir / "final"))
    logger.info("Training complete!")
