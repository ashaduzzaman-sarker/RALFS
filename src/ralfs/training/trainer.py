# src/ralfs/training/trainer.py
from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
from ralfs.core.logging import get_logger
from ralfs.training.dataset import FiDDataset
from ralfs.evaluation.metrics import evaluate_predictions

logger = get_logger(__name__)

def train(cfg):
    accelerator = Accelerator(
        mixed_precision=cfg.training.mixed_precision,
        gradient_accumulation_steps=cfg.model.grad_accum
    )

    # Tokenizer & model
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

    # Datasets
    train_dataset = FiDDataset(cfg, split="train")
    val_dataset = FiDDataset(cfg, split="validation")

    train_loader = DataLoader(train_dataset, batch_size=cfg.model.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.model.lr, weight_decay=cfg.model.weight_decay)

    # Prepare with accelerator
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    total_steps = len(train_loader) * cfg.model.epochs
    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_main_process)

    best_rouge = 0.0
    step = 0

    for epoch in range(cfg.model.epochs):
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                inputs = tokenizer(
                    batch["inputs"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(accelerator.device)
                
                labels = tokenizer(
                    batch["summaries"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).input_ids.to(accelerator.device)

                loss = model(**inputs, labels=labels).loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                step += 1

                if step % cfg.training.logging_steps == 0:
                    logger.info(f"Step {step} | Loss: {loss.item():.4f}")

                if step % cfg.training.eval_steps == 0:
                    model.eval()
                    predictions = []
                    references = []
                    for val_batch in val_loader:
                        with torch.no_grad():
                            generated = model.generate(
                                val_batch["input_ids"].to(accelerator.device),
                                max_length=200
                            )
                        pred = tokenizer.decode(generated[0], skip_special_tokens=True)
                        predictions.append({"summary": pred})
                        references.append({"summary": val_batch["summary"]})
                    
                    scores = evaluate_predictions(predictions, references)
                    if scores["rouge2"] > best_rouge:
                        best_rouge = scores["rouge2"]
                        accelerator.save_state(f"{cfg.training.output_dir}/best")

                    model.train()

        accelerator.save_state(f"{cfg.training.output_dir}/epoch_{epoch+1}")

    logger.info("Training complete!")
    accelerator.save_state(f"{cfg.training.output_dir}/final")
