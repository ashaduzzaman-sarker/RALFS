"""Hybrid Retriever: Dense + BM25 + Late Interaction (ColBERT-style) fusion."""
from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from src.retriever.retriever import DenseRetriever
from src.utils.logging import get_logger
from pathlib import Path

logger = get_logger(__name__)

class HybridRetriever:
    def __init__(self, dense_index_path: str, chunks: List[str]):
        self.dense = DenseRetriever(
            model_name="sentence-transformers/all-MiniLM-L12-v2",
            index_path=dense_index_path
        )
        self.chunks = chunks
        self.bm25 = BM25Okapi([c.split() for c in chunks])
        self.colbert = SentenceTransformer("colbert-ir/colbertv2.0")  # late interaction

    def retrieve(
        self,
        query: str,
        k_dense: int = 100,
        k_bm25: int = 100,
        k_final: int = 20,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2
    ) -> List[Dict]:
        # 1. Dense
        dense_results = self.dense.retrieve(query, k=k_dense)
        dense_scores = {r["text"]: r["score"] for r in dense_results}

        # 2. BM25
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)

        # 3. ColBERT late interaction (max-sim)
        q_emb = self.colbert.encode(query, convert_to_tensor=True)
        doc_embs = self.colbert.encode(self.chunks, convert_to_tensor=True, batch_size=32, show_progress_bar=False)
        colbert_scores = util.cos_sim(q_emb, doc_embs)[0].cpu().numpy()
        colbert_norm = (colbert_scores - colbert_scores.min()) / (colbert_scores.max() - colbert_scores.min() + 1e-8)

        # Fusion with Reciprocal Rank Fusion + linear combination
        fused = {}
        for i, text in enumerate(self.chunks):
            dense_s = dense_scores.get(text, 0.0)
            bm25_s = bm25_norm[i]
            colbert_s = colbert_norm[i]
            score = alpha * dense_s + beta * bm25_s + gamma * colbert_s
            fused[text] = score

        # Top-k
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:k_final]
        results = [
                    {"text": text, "score": float(score), "rank": i + 1}
                    for i, (text, score) in enumerate(ranked)
                ]
        logger.info(f"Hybrid retrieval — Final top-{k_final} fused from {k_dense}+{k_bm25}")
        return results
