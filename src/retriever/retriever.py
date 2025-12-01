"""Dense Retriever with FAISS."""
from __future__ import annotations
import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from src.retriever.encoder import RALFSEncoder
from src.utils.io import RALFSDataManager
from src.utils.logging import get_logger
from pathlib import Path

logger = get_logger(__name__)

class DenseRetriever:
    def __init__(self, model_name: str, index_path: str):
        self.encoder = RALFSEncoder(model_name)
        self.index_path = Path(index_path)
        self.index = None
        self.chunks = None
        self.load_index()

    def load_index(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        metadata_path = self.index_path.with_suffix('.metadata.json')
        data = RALFSDataManager.load_json(metadata_path)
        self.chunks = data["texts"]
        logger.info(f"Loaded FAISS index with {self.index.ntotal} passages")

    def retrieve(self, query: str, k: int = 20) -> List[Dict]:
        q_emb = self.encoder.encode([query])
        scores, indices = self.index.search(q_emb.astype('float32'), k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            results.append({
                "text": self.chunks[idx],
                "score": float(score),
                "rank": len(results) + 1
            })
        return results