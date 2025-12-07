# src/ralfs/training/trainer.py
from __future__ import annotations
from typing import List, Dict, Tuple
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from accelerate import Accelerator
from tqdm.auto import tqdm
from ralfs.core.config import RALFSConfig
from ralfs.core.logging import get_logger
from peft import get_peft_model, LoraConfig, TaskType

logger = get_logger(__name__)

class FiDTrainer:
    def __init__(self, cfg: RALFSConfig):
        self.cfg = cfg.train
        self.accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=self.cfg.grad_accum)
        self.device = self.accelerator.device

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model.name)

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q", "v"],
            bias="none"
        )
        self.model = get_peft_model(self.model, lora_config)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        logger.info(f"FiD Trainer ready: {self.cfg.model.name} on {self.device}")

    def train(self, dataloader: DataLoader):
        for epoch in range(self.cfg.epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(dataloader):
                queries, summaries, passages_list = batch
                inputs = []
                for q, passages in zip(queries, passages_list):
                    ctx = " [SEP] ".join([p["text"][:500] for p in passages[:20]])
                    inputs.append(f"question: {q} context: {ctx}")

                encoded = self.tokenizer(
                    inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.cfg.model.max_input_length
                ).to(self.device)

                labels = self.tokenizer(
                    summaries,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.cfg.model.max_output_length
                ).input_ids.to(self.device)

                with self.accelerator.accumulate(self.model):
                    loss = self.model(**encoded, labels=labels).loss
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{self.cfg.epochs} | Loss: {avg_loss:.4f}")

        self.accelerator.save_state(self.cfg.training.output_dir)
        logger.info("Training complete! Checkpoints saved to checkpoints/arxiv_fid")