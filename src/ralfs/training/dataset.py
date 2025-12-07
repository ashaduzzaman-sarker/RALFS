# src/ralfs/training/dataset.py
from __future__ import annotations
from typing import Tuple
from torch.utils.data import Dataset
from datasets import load_dataset
from ralfs.retriever.factory import create_retriever
from ralfs.utils.io import load_json
from ralfs.core.logging import get_logger
from ralfs.core.config import RALFSConfig

logger = get_logger(__name__)

class FiDDataset(Dataset):
    def __init__(self, cfg: RALFSConfig, split: str = "train"):
        self.cfg = cfg
        self.data = load_dataset(self.cfg.data.dataset, split=split)
        logger.info(f"Loaded {len(self.data)} samples from {self.cfg.data.dataset} ({split})")

        meta = load_json(self.cfg.retriever.chunks_path, as_jsonl=True)
        chunks = [c["text"] for c in meta]
        self.retriever = create_retriever(self.cfg)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, str, List[Dict]]:
        item = self.data[idx]
        query = item["article"][:2000]
        summary = item["abstract"]
        passages = self.retriever.retrieve(query)
        return query, summary, passages