# src/ralfs/data/downloader.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from datasets import load_dataset, Dataset
from pathlib import Path
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