from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from .constants import ROOT_DIR

@dataclass
class RALFSConfig:
    task: str = "train"
    data: DictConfig = None  # type: ignore
    retriever: DictConfig = None  # type: ignore
    generator: DictConfig = None  # type: ignore
    train: DictConfig = None  # type: ignore

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "RALFSConfig":
        return cls(
            task=cfg.get("task", "train"),
            data=cfg.get("data", OmegaConf.create({})),
            retriever=cfg.get("retriever", OmegaConf.create({})),
            generator=cfg.get("generator", OmegaConf.create({})),
            train=cfg.get("train", OmegaConf.create({})),
        )

def load_config(config_path: str | Path = ROOT_DIR / "configs" / "ralfs.yaml") -> RALFSConfig:
    cfg = OmegaConf.load(config_path)
    return RALFSConfig.from_hydra(cfg)