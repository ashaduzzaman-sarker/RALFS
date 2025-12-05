# src/ralfs/retriever/reranker.py
from __future__ import annotations
from typing import List, Dict
from sentence_transformers import CrossEncoder
from ralfs.core.logging import get_logger

logger = get_logger(__name__)

class CrossEncoderReranker:
    def __init__(self, cfg):
        self.cfg = cfg.retriever.reranker
        self.model = CrossEncoder(self.cfg.model, max_length=512)
        logger.info(f"Reranker loaded: {self.cfg.model}")

    def rerank(self, query: str, candidates: List[Dict], top_k: int | None = None) -> List[Dict]:
        top_k = top_k or self.cfg.top_k
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.predict(pairs, batch_size=32)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {**doc, "score": float(score), "rank": i+1}
            for i, (doc, score) in enumerate(ranked)
        ]