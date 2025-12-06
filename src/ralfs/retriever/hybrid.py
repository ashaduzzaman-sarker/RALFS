# src/ralfs/retriever/hybrid.py
from __future__ import annotations
from typing import List, Dict
from ralfs.retriever.dense import DenseRetriever
from ralfs.retriever.sparse import SparseRetriever
from ralfs.retriever.colbert import ColbertRetriever
from ralfs.retriever.reranker import CrossEncoderReranker
from ralfs.retriever.utils import reciprocal_rank_fusion
from ralfs.core.logging import get_logger

logger = get_logger(__name__)

class HybridRetriever:
    def __init__(self, cfg):
        self.cfg = cfg.retriever
        self.dense = DenseRetriever(cfg)
        self.colbert = ColbertRetriever(cfg)
        self.reranker = CrossEncoderReranker(cfg)

        # Load chunks once
        from ralfs.utils.io import load_json
        chunks = load_json(self.cfg.chunks_path, as_jsonl=True)
        self.chunks = [c["text"] for c in chunks]
        self.sparse = SparseRetriever(cfg, self.chunks)

        # Load dense + colbert index
        self.dense.load_index(cfg)
        self.colbert.load_index(cfg)

    def retrieve(self, query: str) -> List[Dict]:
        # Safe k values
        k_dense = min(self.cfg.dense.k, len(self.chunks))
        k_sparse = min(self.cfg.sparse.k, len(self.chunks))
        k_colbert = min(self.cfg.colbert.k, len(self.chunks))
        
        dense = self.dense.retrieve(query, k=k_dense)
        sparse = self.sparse.retrieve(query, k=k_sparse)
        colbert = self.colbert.retrieve(query, k=k_colbert)  # Now safe
        
        # Rest unchanged
        fused = reciprocal_rank_fusion([dense, sparse, colbert])
        final = self.reranker.rerank(query, fused, top_k=min(self.cfg.reranker.top_k, len(fused)))
        
        return final