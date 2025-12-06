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
    """
    Lightweight ColBERT-style late interaction using sentence-transformers.
    Uses token-level max-sim — real late interaction, no colbert-ai needed.
    """
    def __init__(self, cfg):
        self.cfg = cfg.retriever.colbert
        self.model = SentenceTransformer(self.cfg.model)
        self.model.eval()
        self.doc_token_embs = None
        self.chunks = None

    def load_index(self, cfg) -> None:
        chunks_path = Path(cfg.retriever.chunks_path)
        chunks = load_json(chunks_path, as_jsonl=True)
        self.chunks = [c["text"] for c in chunks]

        cache_path = Path(cfg.retriever.index_path).with_suffix(".colbert_token.pt")
        if cache_path.exists():
            data = torch.load(cache_path)
            self.doc_token_embs = data["embs"]
            logger.info(f"Loaded cached ColBERT token embeddings")
        else:
            logger.info("Building ColBERT token embeddings (one-time)...")
            # Encode each document into token embeddings
            all_token_embs = []
            for text in self.chunks:
                encoded = self.model.encode(
                    text,
                    convert_to_tensor=True,
                    output_value='token_embeddings',
                    show_progress_bar=False
                )
                all_token_embs.append(encoded)
            self.doc_token_embs = all_token_embs
            torch.save({"embs": all_token_embs}, cache_path)
            logger.info("ColBERT token cache saved")

    def retrieve(self, query: str, k: int | None = None) -> List[Dict]:
        k = k or self.cfg.k
        if self.doc_token_embs is None:
            raise RuntimeError("Call load_index() first")

        # Encode query tokens
        q_encoded = self.model.encode(
            query,
            convert_to_tensor=True,
            output_value='token_embeddings'
        )  # [n_q_tokens, dim]

        scores = []
        for doc_emb in self.doc_token_embs:
            # Late interaction: max sim per query token
            sims = util.cos_sim(q_encoded, doc_emb)  # [n_q, n_d]
            max_sim_per_q = sims.max(dim=1).values        # [n_q]
            score = max_sim_per_q.mean().item()           # scalar
            scores.append(score)

        # Top-k
        top_indices = torch.topk(torch.tensor(scores), k=min(k, len(scores))).indices.tolist()

        return [
            {
                "text": self.chunks[i],
                "score": scores[i],
                "rank": rank
            }
            for rank, i in enumerate(top_indices, 1)
        ]