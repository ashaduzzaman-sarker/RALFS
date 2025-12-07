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
    
