# src/training/trainer.py
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from accelerate import Accelerator
from tqdm.auto import tqdm
from src.utils.logging import get_logger

logger = get_logger(__name__)

class FiDTrainer:
    def __init__(self, model_name="google/flan-t5-large", lr=3e-5, batch_size=4, grad_accum=8):
        self.accelerator = Accelerator(mixed_precision="bf16")
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.grad_accum = grad_accum

        # Prepare with accelerator
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )
        self.device = self.accelerator.device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(dataloader, disable=not self.accelerator.is_main_process)):
            queries, summaries, passages_list = batch

            inputs = []
            for q, passages in zip(queries, passages_list):
                ctx = " [SEP] ".join([p.get("text", "")[:500] for p in passages[:20]])
                inputs.append(f"question: {q} context: {ctx}")

            encodings = self.tokenizer(
                inputs, padding=True, truncation=True, max_length=1024, return_tensors="pt"
            ).to(self.device)

            labels = self.tokenizer(
                summaries, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).input_ids.to(self.device)

            loss = self.model(input_ids=encodings.input_ids, labels=labels).loss
            loss = loss / self.grad_accum
            self.accelerator.backward(loss)
            total_loss += loss.item() * self.grad_accum

            if (step + 1) % self.grad_accum == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        return total_loss / len(dataloader)
