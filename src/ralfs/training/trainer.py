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
from ralfs.evaluation.metrics import evaluate_predictions

logger = get_logger(__name__)

def train(cfg):
    accelerator = Accelerator(
        mixed_precision=cfg.training.mixed_precision,
        gradient_accumulation_steps=cfg.model.grad_accum
    )

    # Safe output directory
    dataset_name = cfg.data.dataset
    output_dir = Path("checkpoints") / f"{dataset_name}_fid"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Training checkpoints → {output_dir}")

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.name)

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q", "v"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Datasets
    train_dataset = FiDDataset(cfg, split="train")
    val_dataset = FiDDataset(cfg, split="validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        num_workers=0,  # Safe for Colab
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay
    )

    # Prepare with accelerator
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    total_steps = len(train_loader) * cfg.model.epochs
    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_main_process)

    best_rouge = 0.0
    global_step = 0

    for epoch in range(cfg.model.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            with accelerator.accumulate(model):
                # Tokenize inputs
                inputs = tokenizer(
                    batch["inputs"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                )
                inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

                # Tokenize labels
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
                epoch_loss += loss.item()
                progress_bar.update(1)
                global_step += 1

                if global_step % cfg.training.logging_steps == 0:
                    logger.info(f"Epoch {epoch+1} | Step {global_step} | Loss: {loss.item():.4f}")

                # Validation
                if global_step % cfg.training.eval_steps == 0:
                    model.eval()
                    predictions = []
                    references = []
                    for val_batch in val_loader:
                        with torch.no_grad():
                            input_ids = tokenizer(
                                val_batch["inputs"],
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=1024
                            ).input_ids.to(accelerator.device)

                            generated = model.generate(
                                input_ids,
                                max_length=200,
                                num_beams=4,
                                early_stopping=True
                            )
                            pred = tokenizer.decode(generated[0], skip_special_tokens=True)
                            predictions.append({"summary": pred})
                            references.append({"summary": val_batch["summaries"][0]})

                    scores = evaluate_predictions(predictions, references)
                    rouge2 = scores["rouge2"]

                    if rouge2 > best_rouge:
                        best_rouge = rouge2
                        accelerator.save_state(str(output_dir / "best"))
                        logger.info(f"New best ROUGE-2: {rouge2:.4f} → saved 'best' checkpoint")

                    model.train()

        # Save epoch checkpoint
        accelerator.save_state(str(output_dir / f"epoch_{epoch+1}"))

    # Final save
    accelerator.save_state(str(output_dir / "final"))
    logger.info(f"Training complete! Final model: {output_dir / 'final'}")
