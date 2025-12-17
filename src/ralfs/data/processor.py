# src/ralfs/data/__init__.py
from .processor import run_preprocessing
from .indexer import build_index
__all__ = [
    "run_preprocessing",
    "build_index",
]

# src/ralfs/data/chunker.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import nltk
import re
from ralfs.core.logging import get_logger

logger = get_logger(__name__)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

@dataclass
class Chunk:
    text: str
    chunk_id: str
    doc_id: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        return asdict(self)

class SemanticChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = nltk.sent_tokenize

    def _clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.strip())

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        text = self._clean_text(text)
        sentences = self.tokenizer(text)
        chunks: List[Chunk] = []
        current: List[str] = []
        current_tokens = 0
        start_pos = 0

        for sent in sentences:
            sent_tokens = len(sent.split())
            if current_tokens + sent_tokens > self.chunk_size and len(current) > 2:
                chunk_text = " ".join(current)
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=f"{doc_id}_c{len(chunks)}",
                    doc_id=doc_id,
                    start_char=start_pos,
                    end_char=start_pos + len(chunk_text),
                    metadata={}
                ))
                # Overlap
                overlap_text = " ".join(current[-2:])
                current = current[-2:]
                current_tokens = len(overlap_text.split())
                start_pos += len(chunk_text) - len(overlap_text)
            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunk_text = " ".join(current)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"{doc_id}_c{len(chunks)}",
                doc_id=doc_id,
                start_char=start_pos,
                end_char=len(text),
                metadata={}
            ))

        logger.info(f"Chunked {doc_id} → {len(chunks)} chunks")
        return chunks
# src/ralfs/data/downloader.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from datasets import load_dataset, Dataset
from pathlib import Path
import json
from ralfs.core.logging import get_logger
from ralfs.core.constants import RAW_DIR

logger = get_logger(__name__)

@dataclass
class Document:
    id: str
    text: str
    summary: str
    title: str
    domain: str
    source: str

class DatasetDownloader:
    SUPPORTED = {
        "arxiv": {
            "hf_name": "ccdv/arxiv-summarization",
            "config": "document",
            "text_key": "article",
            "summary_key": "abstract",
            "domain": "scientific"
        },
        "govreport": {
            "hf_name": "ccdv/govreport-summarization",
            "config": None,
            "text_key": "report",
            "summary_key": "summary",
            "domain": "government"
        },
        "booksum": {
            "hf_name": "kmfoda/booksum",
            "config": None,
            "text_key": "chapter",
            "summary_key": "summary",
            "domain": "literature"
        }
    }

    @classmethod
    def download(
        cls,
        dataset_name: str,
        split: str = "train",
        max_samples: int | None = None
    ) -> List[Document]:
        if dataset_name not in cls.SUPPORTED:
            raise ValueError(f"Dataset {dataset_name} not supported. Choose from: {list(cls.SUPPORTED)}")

        cfg = cls.SUPPORTED[dataset_name]
        logger.info(f"Downloading {dataset_name.upper()} ({split}) from {cfg['hf_name']}")

        dataset = load_dataset(cfg["hf_name"], cfg["config"], split=split) if cfg["config"] else load_dataset(cfg["hf_name"], split=split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        docs = []
        for i, ex in enumerate(dataset):
            docs.append(Document(
                id=f"{dataset_name}_{split}_{i}",
                text=str(ex[cfg["text_key"]]),
                summary=str(ex[cfg["summary_key"]]),
                title=ex.get("title", f"Doc {i}"),
                domain=cfg["domain"],
                source=dataset_name
            ))
        logger.info(f"Downloaded {len(docs)} documents")
        return docs

    @classmethod
    def save(
        cls,
        docs: List[Document],
        dataset_name: str,
        split: str = "train"
    ) -> Path:
        """Save documents to data/raw directory as JSONL."""
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        
        output_path = RAW_DIR / f"{dataset_name}_{split}.jsonl"
        logger.info(f"Saving {len(docs)} documents to {output_path}")
        
        with open(output_path, "w") as f:
            for doc in docs:
                f.write(json.dumps(asdict(doc)) + "\n")
        
        logger.info(f"Saved {len(docs)} documents to {output_path}")
        return output_path

    @classmethod
    def download_and_save(
        cls,
        dataset_name: str,
        split: str = "train",
        max_samples: int | None = None
    ) -> Path:
        """Download dataset and save to data/raw."""
        docs = cls.download(dataset_name, split, max_samples)
        return cls.save(docs, dataset_name, split)
    

# src/ralfs/data/indexer.py
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ralfs.core.logging import get_logger
from ralfs.utils.io import save_json, load_json
from ralfs.core.constants import INDEX_DIR, PROCESSED_DIR

logger = get_logger(__name__)

def build_index(cfg) -> None:
    chunks_path = Path(cfg.retriever.chunks_path)
    chunks = load_json(chunks_path, as_jsonl=True)
    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(cfg.retriever.dense.model)
    logger.info(f"Encoding {len(texts)} chunks with {cfg.retriever.dense.model}")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEX_DIR / "faiss.index"
    faiss.write_index(index, str(index_path))

    save_json({"texts": texts}, index_path.with_suffix(".metadata.json"))
    logger.info(f"FAISS index saved: {index_path}")
# src/ralfs/data/processor.py
from __future__ import annotations
from typing import List
from pathlib import Path
from ralfs.core.logging import get_logger
from ralfs.core.constants import PROCESSED_DIR, RAW_DIR
from ralfs.data.downloader import DatasetDownloader
from ralfs.data.chunker import SemanticChunker
from ralfs.utils.io import save_json

logger = get_logger(__name__)

def run_preprocessing(cfg) -> None:
    dataset_name = cfg.data.dataset
    max_samples = cfg.data.get("max_samples", None)

    logger.info(f"Starting preprocessing for {dataset_name}")

    # 1. Download
    docs = DatasetDownloader.download(
        dataset_name=dataset_name,
        split="train",
        max_samples=max_samples
    )

    # 2. Chunk
    chunker = SemanticChunker(
        chunk_size=cfg.data.chunk_size,
        overlap=cfg.data.overlap
    )
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk(doc.text, doc.id)
        for c in chunks:
            c.metadata.update({
                "doc_id": doc.id,
                "title": doc.title,
                "summary": doc.summary,
                "domain": doc.domain
            })
        all_chunks.extend(chunks)

    # 3. Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{dataset_name}_chunks.jsonl"
    save_json([chunk.to_dict() for chunk in all_chunks], out_path, as_jsonl=True)

    logger.info(f"Saved {len(all_chunks)} chunks → {out_path}")