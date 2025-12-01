"""src/data/dataset.py
RALFS PyTorch Datasets."""
from __future__ import annotations

from torch.utils.data import Dataset
from typing import List, Dict, Any
import torch
from src.utils.io import RALFSDataManager


class RALFSChunkDataset(Dataset):
    """Dataset for preprocessed chunks."""
    
    def __init__(self, chunks_path: str):
        self.chunks = RALFSDataManager.load_json(chunks_path)
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        chunk = self.chunks[idx]
        return {
            "chunk_id": chunk["id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"]
        }


class RALFSTrainDataset(Dataset):
    """Training dataset for FiD with retrieved evidence."""
    
    def __init__(self, chunks_path: str, summaries_path: str):
        self.chunks = RALFSDataManager.load_json(chunks_path)
        self.summaries = RALFSDataManager.load_json(summaries_path)
        self.paired_data = self._pair_data()
    
    def _pair_data(self) -> List[Dict[str, Any]]:
        # Simplified pairing for now (will enhance with retriever)
        paired = []
        for chunk in self.chunks[:1000]:  # Experiment size
            doc_id = chunk["metadata"]["doc_id"]
            summary = next((s["summary"] for s in self.summaries 
                          if s["id"].startswith(doc_id)), "")
            paired.append({
                "context": chunk["text"],
                "summary": summary
            })
        return paired
    
    def __len__(self) -> int:
        return len(self.paired_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.paired_data[idx]
        return {
            "input_ids": item["context"],  # Will be tokenized
            "labels": item["summary"]
        }
