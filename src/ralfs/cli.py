# src/ralfs/cli.py
from __future__ import annotations
import hydra
from omegaconf import DictConfig
from ralfs.core.logging import get_logger

log = get_logger()

@hydra.main(config_path="../configs", config_name="ralfs", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("RALFS v1.0 — Starting task: [bold green]%s[/]", cfg.task)

    if cfg.task == "preprocess":
        from ralfs.data.processor import run_preprocessing
        run_preprocessing(cfg)
    elif cfg.task == "build_index":
        from ralfs.data.indexer import build_index
        build_index(cfg)
    elif cfg.task == "train":
        from ralfs.training.trainer import train
        train(cfg)
    elif cfg.task == "generate":
        from ralfs.generator.fid import generate
        generate(cfg)
    elif cfg.task == "evaluate":
        from ralfs.evaluation.main import evaluate
        evaluate(cfg)
    else:
        log.error("Unknown task: %s", cfg.task)
        raise ValueError(f"Task must be one of: preprocess, build_index, train, generate, evaluate")

if __name__ == "__main__":
    main()