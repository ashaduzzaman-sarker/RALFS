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

def rank_normalize(scores: List[float]) -> List[float]:
    """Rank-based normalization: 1/(rank+60) — standard in hybrid search"""
    ranks = np.argsort(np.argsort(scores))[::-1] + 1  # 1 = best
    return [1.0 / (60 + r) for r in ranks]

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
        k = 100  # overfetch
        
        # 1. Retrieve from each
        dense_results = self.dense.retrieve(query, k=k)
        sparse_results = self.sparse.retrieve(query, k=k)
        colbert_results = self.colbert.retrieve(query, k=k)

        # 2. Rank-based fusion (RRF) — this is the GOLD STANDARD
        all_results = [dense_results, sparse_results, colbert_results]
        fused = {}
        for rank_offset, results in enumerate(all_results):
            for rank, doc in enumerate(results):
                text = doc["text"]
                if text not in fused:
                    fused[text] = 0
                # RRF formula
                fused[text] += 1 / (60 + rank + 1)

        # 3. Top candidates for reranking
        top_candidates = [
            {"text": text, "score": score}
            for text, score in sorted(fused.items(), key=lambda x: -x[1])[:50]
        ]

        # 4. Final reranking
        final = self.reranker.rerank(query, top_candidates, top_k=self.cfg.reranker.top_k)

        logger.info(f"Hybrid RRF + Rerank → Retrieved top-{len(final)} for '{query}'")
        return final