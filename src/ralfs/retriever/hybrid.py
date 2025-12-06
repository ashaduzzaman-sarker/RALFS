# src/ralfs/retriever/hybrid.py
from __future__ import annotations
from typing import List, Dict
import numpy as np
from ralfs.retriever.dense import DenseRetriever
from ralfs.retriever.sparse import SparseRetriever
from ralfs.retriever.colbert import ColbertRetriever
from ralfs.retriever.reranker import CrossEncoderReranker
from ralfs.core.logging import get_logger

logger = get_logger(__name__)

class HybridRetriever:
    def __init__(self, cfg):
        self.cfg = cfg.retriever
        
        # Load chunks
        from ralfs.utils.io import load_json
        chunks = load_json(self.cfg.chunks_path, as_jsonl=True)
        self.chunks = [c["text"] for c in chunks]

        # Initialize retrievers
        self.dense = DenseRetriever(cfg)
        self.sparse = SparseRetriever(cfg, self.chunks)
        self.colbert = ColbertRetriever(cfg)
        self.reranker = CrossEncoderReranker(cfg)

        self.dense.load_index(cfg)
        self.colbert.load_index(cfg)

    def retrieve(self, query: str) -> List[Dict]:
        # 1. Retrieve from each
        dense = self.dense.retrieve(query, k=self.cfg.dense.k)
        sparse = self.sparse.retrieve(query, k=self.cfg.sparse.k)
        colbert = self.colbert.retrieve(query, k=self.cfg.colbert.k)

        # 2. Normalize and fuse
        def normalize(scores: List[float]) -> List[float]:
            s = np.array(scores)
            return (s - s.min()) / (s.max() - s.min() + 1e-8)

        fused = {}
        for r in dense:
            fused[r["text"]] = fused.get(r["text"], 0) + self.cfg.fusion.alpha * normalize([r["score"]])[0]
        for r in sparse:
            fused[r["text"]] = fused.get(r["text"], 0) + self.cfg.fusion.beta * normalize([r["score"]])[0]
        for r in colbert:
            fused[r["text"]] = fused.get(r["text"], 0) + self.cfg.fusion.gamma * normalize([r["score"]])[0]

        # 3. Top candidates
        top_candidates = [
            {"text": text, "score": score}
            for text, score in sorted(fused.items(), key=lambda x: -x[1])[:50]
        ]

        # 4. Rerank and normalize final scores to [0,1]
        final = self.reranker.rerank(query, top_candidates, top_k=self.cfg.reranker.top_k)
        final_scores = [r["score"] for r in final]
        norm_scores = normalize(final_scores)
        for i, r in enumerate(final):
            r["score"] = norm_scores[i]

        logger.info(f"Hybrid + Rerank → Retrieved top-{len(final)} for '{query}'")
        return final