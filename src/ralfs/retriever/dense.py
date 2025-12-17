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