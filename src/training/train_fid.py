# src/training/train_fid.py
import hydra
from omegaconf import DictConfig
import torch
import torch.multiprocessing as mp
from src.training.trainer import FiDTrainer
from src.training.fid_dataset import FiDDataset
from torch.utils.data import DataLoader
from src.utils.logging import get_logger

# ← CRITICAL: Fix CUDA + multiprocessing crash
mp.set_start_method('spawn', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')

logger = get_logger(__name__)

@hydra.main(config_path="../../configs/train", config_name="fid_train", version_base=None)
def main(cfg: DictConfig):
    logger.info("Starting FiD training on arXiv-summarization (T4-optimized)")

    train_dataset = FiDDataset(split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        num_workers=0,           # ← Critical: prevents CUDA fork crash
        pin_memory=True
    )

    trainer = FiDTrainer(
        model_name=cfg.model.name,
        lr=cfg.model.lr,
        batch_size=cfg.model.batch_size,
        grad_accum=cfg.model.grad_accum
    )

    for epoch in range(cfg.model.epochs):
        loss = trainer.train_epoch(train_loader)
        logger.info(f"Epoch {epoch+1}/{cfg.model.epochs} | Loss: {loss:.4f}")

        if (epoch + 1) % 2 == 0:
            trainer.accelerator.save_state(f"{cfg.output_dir}/epoch_{epoch+1}")

    trainer.accelerator.save_state(cfg.output_dir)
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
