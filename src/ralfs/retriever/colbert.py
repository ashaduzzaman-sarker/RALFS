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