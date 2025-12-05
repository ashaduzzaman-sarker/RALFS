# src/ralfs/retriever/dense.py
from __future__ import annotations
from typing import List, Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from ralfs.retriever.base import BaseRetriever
from ralfs.core.logging import get_logger
from ralfs.utils.io import load_json

logger = get_logger(__name__)

class DenseRetriever(BaseRetriever):
    def __init__(self, cfg):
        self.cfg = cfg.retriever.dense
        self.model = SentenceTransformer(self.cfg.model)
        self.model.eval()
        self.index = None
        self.chunks = None

    def load_index(self, cfg) -> None:
        index_path = Path(cfg.retriever.index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        self.index = faiss.read_index(str(index_path))
        meta = load_json(index_path.with_suffix(".metadata.json"))
        self.chunks = meta["texts"]
        logger.info(f"DenseRetriever loaded {len(self.chunks)} passages")

    def retrieve(self, query: str, k: int | None = None) -> List[Dict]:
        k = k or self.cfg.k
        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(q_emb.astype('float32'), k * 3)  # overfetch

        results = []
        seen = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
            text = self.chunks[idx]
            if text in seen:
                continue
            seen.add(text)
            results.append({
                "text": text,
                "score": float(score),
                "rank": len(results) + 1
            })
            if len(results) >= k:
                break
        return results