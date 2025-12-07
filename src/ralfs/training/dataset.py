# src/ralfs/training/dataset.py
from __future__ import annotations
from torch.utils.data import Dataset
from typing import List, Dict
from ralfs.data.downloader import DatasetDownloader
from ralfs.retriever.factory import create_retriever
from ralfs.core.logging import get_logger

logger = get_logger(__name__)

class FiDDataset(Dataset):
    def __init__(self, cfg, split: str = "train"):
        self.cfg = cfg
        self.split = split
        
        # Load raw documents (with gold summaries)
        self.documents = DatasetDownloader.download(
            dataset_name=cfg.data.dataset,
            split=split,
            max_samples=cfg.data.get("max_samples")
        )
        
        # Create retriever for passage retrieval
        self.retriever = create_retriever(cfg)
        
        logger.info(f"FiDDataset ready: {len(self.documents)} documents, {split} split")

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> Dict:
        doc = self.documents[idx]
        
        # Retrieve top passages for this document
        passages = self.retriever.retrieve(doc.text[:2000])  # Use beginning as query
        
        return {
            "query": doc.text[:2000],
            "summary": doc.summary,
            "passages": passages[:20]  # Limit to top-20
        }