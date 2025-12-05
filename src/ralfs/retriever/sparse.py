# src/ralfs/retriever/sparse.py
from __future__ import annotations
from typing import List, Dict
from rank_bm25 import BM25Okapi
from ralfs.retriever.base import BaseRetriever
from ralfs.core.logging import get_logger

logger = get_logger(__name__)

class SparseRetriever(BaseRetriever):
    def __init__(self, cfg, chunks: List[str]):
        self.cfg = cfg.retriever.sparse
        self.chunks = chunks
        self.bm25 = BM25Okapi([doc.split() for doc in chunks])
        logger.info(f"SparseRetriever (BM25) ready with {len(chunks)} docs")

    def load_index(self, cfg) -> None:
        pass

    def retrieve(self, query: str, k: int | None = None) -> List[Dict]:
        k = k or self.cfg.k
        scores = self.bm25.get_scores(query.split())
        top_idx = scores.argsort()[::-1][:k]
        return [
            {
                "text": self.chunks[i],
                "score": float(scores[i]),
                "rank": rank
            }
            for rank, i in enumerate(top_idx, 1)
        ]