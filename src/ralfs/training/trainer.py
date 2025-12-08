# src/ralfs/training/trainer.py
from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from accelerate import Accelerator
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model, TaskType
from ralfs.core.logging import get_logger

logger = get_logger(__name__)

class FiDTrainer:
    def __init__(self, cfg):
        self.cfg = cfg.train
        self.accelerator = Accelerator(
            mixed_precision="fp16",
            gradient_accumulation_steps=self.cfg.model.grad_accum
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model.name)
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"],
            bias="none"
        )
        self.model = get_peft_model(self.model, lora_config)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.model.lr)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        
        logger.info(f"FiD Trainer ready | Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, disable=not self.accelerator.is_main_process):
            queries = batch["query"]
            summaries = batch["summary"]
            passages_list = batch["passages"]
            
            inputs = []
            for q, passages in zip(queries, passages_list):
                texts = []
                for p in passages:
                    if isinstance(p, dict):
                        texts.append(p.get("text", "")[:400])
                    elif isinstance(p, str):
                        texts.append(p[:400])
                    else:
                        texts.append("")
                ctx = " [SEP] ".join(texts)
                inputs.append(f"question: {q} context: {ctx}")
            
            enc = self.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            )
            labels = self.tokenizer(
                summaries,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).input_ids
            
            enc = {k: v.to(self.accelerator.device) for k, v in enc.items()}
            labels = labels.to(self.accelerator.device)
            
            loss = self.model(**enc, labels=labels).loss
            loss = loss / self.cfg.model.grad_accum
            self.accelerator.backward(loss)
            total_loss += loss.item()
            
            if self.accelerator.sync_gradients:
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return total_loss * self.cfg.model.grad_accum / len(dataloader)

    def train(self, dataloader: DataLoader):
        for epoch in range(self.cfg.model.epochs):
            loss = self.train_epoch(dataloader)
            logger.info(f"Epoch {epoch+1}/{self.cfg.model.epochs} | Loss: {loss:.4f}")
        
        save_path = self.cfg.training.output_dir
        self.accelerator.wait_for_everyone()
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
