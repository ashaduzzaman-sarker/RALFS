"""src/data/download.py
RALFS Dataset Download Utilities."""
from __future__ import annotations

from datasets import load_dataset
from typing import Dict, List, Optional
from src.utils.logging import get_logger
import hydra.utils as hu

from pathlib import Path
logger = get_logger(__name__)



class DatasetDownloader:
    """Download and cache RALFS benchmark datasets."""
    
    SUPPORTED_DATASETS: Dict[str, Dict[str, str]] = {
        "govreport": {
            "name": "csebuetnlp/xlsum", 
            "subset": "govreport"
        },
        "arxiv": {
            "name": "scientific_papers", 
            "subset": "arxiv"
        },
        "booksum": {
            "name": "booksum", 
            "subset": "all"
        },
        "multi_news": {
            "name": "multi_news", 
            "subset": "train"
        }
    }
    
    @classmethod
    def download(cls, dataset_name: str, split: str = "train", 
                 max_samples: Optional[int] = None) -> Dict[str, List[str]]:
        """Download dataset and return document chunks.
        
        Args:
            dataset_name: Dataset identifier
            split: Dataset split (train/val/test)
            max_samples: Maximum samples for experimentation
            
        Returns:
            Dictionary with document texts
        """
        if dataset_name not in cls.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        config = cls.SUPPORTED_DATASETS[dataset_name]
        logger.info(f"Downloading {dataset_name} ({split} split)")
        
        dataset = load_dataset(config["name"], config["subset"], split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        documents = []
        for i, example in enumerate(dataset):
            # Extract full document text
            doc_text = example.get("document", example.get("text", ""))
            if doc_text:
                documents.append({
                    "id": f"{dataset_name}_{split}_{i}",
                    "text": doc_text,
                    "summary": example.get("summary", ""),
                    "title": example.get("title", "")
                })
        
        logger.info(f"Downloaded {len(documents)} documents")
        return {"documents": documents}