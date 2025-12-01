"""Sentence-Transformer encoder for RALFS."""
from __future__ import annotations
from sentence_transformers import SentenceTransformer
from typing import List
import torch

class RALFSEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )