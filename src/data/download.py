"""src/data/download.py
RALFS Dataset Download Utilities."""
from __future__ import annotations

from datasets import load_dataset, Dataset
from typing import Dict, List, Optional, Any
from src.utils.logging import get_logger
from pathlib import Path

logger = get_logger(__name__)


class DatasetDownloader:
    """
    Unified downloader for all RALFS benchmark datasets.
    Supports GovReport, arXiv, BookSum, PubMed, QMSum — all used in top papers.
    """
    
    # FULLY CORRECTED — based on your exact spec
    SUPPORTED_DATASETS: Dict[str, Dict[str, Any]] = {
        "booksum": {
            "hf_name": "kmfoda/booksum",
            "config": None,
            "text_key": "chapter",
            "summary_key": "summary",
            "domain": "literature"
        },
        "arxiv": {
            "hf_name": "ccdv/arxiv-summarization",
            "config": "document",
            "text_key": "article",
            "summary_key": "abstract",
            "domain": "scientific"
        },
        "govreport": {
            "hf_name": "ccdv/govreport-summarization",
            "config": None,  # Works perfectly without config
            "text_key": "report",
            "summary_key": "summary",
            "domain": "government"
        },
        "pubmed": {
            "hf_name": "ccdv/pubmed-summarization",
            "config": "document",
            "text_key": "article",
            "summary_key": "abstract",
            "domain": "biomedical"
        },
        "qmsum": {
            "hf_name": "pszemraj/qmsum-cleaned",
            "config": None,
            "text_key": "input",
            "summary_key": "output",
            "domain": "meeting"
        }
    }

    @classmethod
    def download(
        cls,
        dataset_name: str,
        split: str = "train",
        max_samples: Optional[int] = None
    ) -> Dict[str, List[dict]]:
        """Download and return list of documents with metadata."""
        
        if dataset_name not in cls.SUPPORTED_DATASETS:
            available = ", ".join(cls.SUPPORTED_DATASETS.keys())
            raise ValueError(f"Dataset '{dataset_name}' not supported. Choose from: {available}")

        cfg = cls.SUPPORTED_DATASETS[dataset_name]
        logger.info(f"Downloading {dataset_name.upper()} from {cfg['hf_name']} (split={split})")

        try:
            if cfg["config"] is not None:
                dataset: Dataset = load_dataset(cfg["hf_name"], cfg["config"], split=split)
            else:
                dataset: Dataset = load_dataset(cfg["hf_name"], split=split)
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            raise

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        documents = []
        text_key = cfg["text_key"]
        summary_key = cfg["summary_key"]

        for i, example in enumerate(dataset):
            doc_text = example.get(text_key, "")
            summary = example.get(summary_key, "")
            
            if not doc_text.strip():
                continue  # Skip empty

            documents.append({
                "id": f"{dataset_name}_{split}_{i}",
                "text": str(doc_text),
                "summary": str(summary),
                "title": example.get("title", example.get("book_title", f"Doc {i}")),
                "domain": cfg["domain"],
                "source": dataset_name
            })

        logger.info(f"Successfully downloaded {len(documents)} documents from {dataset_name}")
        return {"documents": documents}