# src/ralfs/retriever/__init__.py
from .base import BaseRetriever
from .dense import DenseRetriever
from .sparse import SparseRetriever
from .hybrid import HybridRetriever
from .reranker import CrossEncoderReranker
from .colbert import ColbertRetriever
from .utils import reciprocal_rank_fusion

__all__ = [
    "BaseRetriever",
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "CrossEncoderReranker",
    "ColbertRetriever",
    "reciprocal_rank_fusion",
]
# src/ralfs/retriever/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict
from pathlib import Path

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 20) -> List[Dict[str, any]]:
        """Return list of {'text': str, 'score': float, 'doc_id': str}"""
        pass

    @abstractmethod
    def load_index(self, index_path: Path) -> None:
        pass
# src/ralfs/retriever/colbert.py
from __future__ import annotations
from typing import List, Dict
from pathlib import Path
from colbert import Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Collection
from ralfs.retriever.base import BaseRetriever
from ralfs.core.logging import get_logger
from ralfs.utils.io import load_json

logger = get_logger(__name__)

class ColbertRetriever(BaseRetriever):
    def __init__(self, cfg):
        self.cfg = cfg.retriever.colbert
        self.searcher = None
        self.collection_path = None

    def load_index(self, cfg) -> None:
        chunks_path = Path(cfg.retriever.chunks_path)
        chunks = load_json(chunks_path, as_jsonl=True)
        texts = [c["text"] for c in chunks]

        # Save collection for ColBERT
        self.collection_path = Path("data/index/collection.tsv")
        self.collection_path.parent.mkdir(exist_ok=True, parents=True)
        with open(self.collection_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(texts):
                f.write(f"{i}\t{text}\n")

        index_name = "ralfs.colbert"
        index_path = Path(f"experiments/ralfs/indexes/{index_name}")

        if not index_path.exists():
            logger.info("Building real ColBERT index (one-time)...")
            with Run().context(RunConfig(nranks=1, experiment="ralfs")):
                config = ColBERTConfig(
                    doc_maxlen=300,
                    nbits=2,
                    kmeans_niters=4
                )
                from colbert import Indexer
                indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=config)
                collection = Collection(path=str(self.collection_path))
                indexer.index(name=index_name, collection=collection, overwrite=True)

        logger.info("Loading real ColBERT searcher...")
        with Run().context(RunConfig(experiment="ralfs")):
            self.searcher = Searcher(index=index_name, collection=str(self.collection_path))
        logger.info(f"Real ColBERT ready with {len(texts)} passages")

    def retrieve(self, query: str, k: int | None = None) -> List[Dict]:
        k = k or self.cfg.k
        pids, ranks, scores = self.searcher.search(query, k=k)
        return [
            {
                "text": self.searcher.collection[pid],
                "score": float(score),
                "rank": rank
            }
            for rank, (pid, score) in enumerate(zip(pids, scores), 1)
        ]
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
        if self.index is None:
            raise RuntimeError("Call load_index() first")

        k = k or self.cfg.k
        k = min(k, self.index.ntotal)  # ← THIS IS THE CORRECT LINE

        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(q_emb.astype('float32'), k)

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
        return results
# src/ralfs/retriever/factory.py
from omegaconf import DictConfig
from .hybrid import HybridRetriever

def create_retriever(cfg: DictConfig) -> HybridRetriever:
    return HybridRetriever(cfg)
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