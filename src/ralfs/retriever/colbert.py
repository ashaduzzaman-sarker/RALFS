# src/ralfs/retriever/colbert.py
from __future__ import annotations
from typing import List, Dict
import torch
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from ralfs.retriever.base import BaseRetriever
from ralfs.core.logging import get_logger
from ralfs.utils.io import load_json

logger = get_logger(__name__)

class ColbertRetriever(BaseRetriever):
    def __init__(self, cfg):
        self.cfg = cfg.retriever.colbert
        self.model = SentenceTransformer(self.cfg.model)
        self.model.eval()
        self.doc_embs = None
        self.chunks = None

    def load_index(self, cfg) -> None:
        chunks_path = Path(cfg.retriever.chunks_path)
        chunks = load_json(chunks_path, as_jsonl=True)
        self.chunks = [c["text"] for c in chunks]

        cache_path = Path(cfg.retriever.index_path).with_suffix(".colbert.pt")
        if cache_path.exists():
            self.doc_embs = torch.load(cache_path)
        else:
            logger.info("Encoding all chunks with ColBERT (one-time cost)...")
            self.doc_embs = self.model.encode(
                self.chunks,
                convert_to_tensor=True,
                batch_size=64,
                show_progress_bar=True
            )
            torch.save(self.doc_embs, cache_path)
        logger.info(f"ColBERT loaded {len(self.chunks)} passages")


    def retrieve(self, query: str, k: int | None = None) -> List[Dict]:
            k = min(k or self.cfg.k, len(self.chunks)) 
            q_emb = self.model.encode(query, convert_to_tensor=True)
            scores = util.cos_sim(q_emb, self.doc_embs)[0]  
            scores = scores.flatten()  
            top_k = torch.topk(scores, k)  
            return [
                {
                    "text": self.chunks[i],
                    "score": float(scores[i]),
                    "rank": rank
                }
                for rank, i in enumerate(top_k.indices, 1)
            ]