# ============================================================================
# File: ralfs/data/downloader.py
# ============================================================================
"""Dataset downloader with support for multiple long-form summarization datasets."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datasets import load_dataset, Dataset
from pathlib import Path
import json
from tqdm import tqdm
from ralfs.core.logging import get_logger
from ralfs.core.constants import RAW_DIR

logger = get_logger(__name__)


@dataclass
class Document:
    """Document data structure."""
    id: str
    text: str
    summary: str
    title: str = ""
    domain: str = ""
    source: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)
    
    def __len__(self) -> int:
        """Return document text length."""
        return len(self.text)


class DatasetDownloader:
    """
    Download and process datasets from HuggingFace.
    
    Supported datasets:
    - arXiv: Scientific papers
    - GovReport: Government reports
    - BookSum: Book chapters
    - PubMed: Medical abstracts
    - Multi-News: Multi-document news
    """
    
    SUPPORTED = {
        "arxiv": {
            "hf_name": "ccdv/arxiv-summarization",
            "config": "document",
            "text_key": "article",
            "summary_key": "abstract",
            "title_key": None,
            "domain": "scientific",
        },
        "govreport": {
            "hf_name": "ccdv/govreport-summarization",
            "config": None,
            "text_key": "report",
            "summary_key": "summary",
            "title_key": None,
            "domain": "government",
        },
        "booksum": {
            "hf_name": "kmfoda/booksum",
            "config": None,
            "text_key": "chapter",
            "summary_key": "summary_text",
            "title_key": "title",
            "domain": "literature",
        },
        "pubmed": {
            "hf_name": "ccdv/pubmed-summarization",
            "config": "document",
            "text_key": "article",
            "summary_key": "abstract",
            "title_key": None,
            "domain": "medical",
        },
        "multi_news": {
            "hf_name": "multi_news",
            "config": None,
            "text_key": "document",
            "summary_key": "summary",
            "title_key": None,
            "domain": "news",
        },
    }
    
    @classmethod
    def list_supported(cls) -> List[str]:
        """List all supported datasets."""
        return list(cls.SUPPORTED.keys())
    
    @classmethod
    def download(
        cls,
        dataset_name: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        use_streaming: bool = False,
    ) -> List[Document]:
        """
        Download dataset from HuggingFace.
        
        Args:
            dataset_name: Name of dataset (e.g., 'arxiv', 'govreport')
            split: Data split ('train', 'validation', 'test')
            max_samples: Maximum number of samples to download (None = all)
            cache_dir: Directory to cache downloaded data
            use_streaming: Use streaming mode for large datasets
        
        Returns:
            List of Document objects
        
        Raises:
            ValueError: If dataset_name is not supported
        """
        if dataset_name not in cls.SUPPORTED:
            supported = ", ".join(cls.list_supported())
            raise ValueError(
                f"Dataset '{dataset_name}' not supported. "
                f"Choose from: {supported}"
            )
        
        cfg = cls.SUPPORTED[dataset_name]
        logger.info(
            f"Downloading {dataset_name.upper()} ({split}) "
            f"from {cfg['hf_name']}"
        )
        
        try:
            # Load dataset
            load_kwargs = {
                "path": cfg["hf_name"],
                "split": split,
                "cache_dir": cache_dir,
                "streaming": use_streaming,
            }
            
            if cfg["config"]:
                load_kwargs["name"] = cfg["config"]
            
            dataset = load_dataset(**load_kwargs)
            
            # Handle streaming vs non-streaming
            if use_streaming:
                if max_samples:
                    dataset = dataset.take(max_samples)
                total = max_samples or "unknown"
            else:
                if max_samples:
                    dataset = dataset.select(range(min(max_samples, len(dataset))))
                total = len(dataset)
            
            logger.info(f"Processing {total} documents...")
            
            # Process documents
            docs = []
            for i, ex in enumerate(tqdm(dataset, desc=f"Loading {dataset_name}")):
                try:
                    # Extract fields
                    text = str(ex.get(cfg["text_key"], ""))
                    summary = str(ex.get(cfg["summary_key"], ""))
                    
                    # Get title if available
                    if cfg["title_key"] and cfg["title_key"] in ex:
                        title = str(ex[cfg["title_key"]])
                    else:
                        title = ex.get("title", f"Document {i}")
                    
                    # Skip empty documents
                    if not text or not summary:
                        logger.warning(f"Skipping document {i}: empty text or summary")
                        continue
                    
                    # Create document
                    doc = Document(
                        id=f"{dataset_name}_{split}_{i}",
                        text=text,
                        summary=summary,
                        title=title,
                        domain=cfg["domain"],
                        source=dataset_name,
                        metadata={
                            "split": split,
                            "index": i,
                            "text_length": len(text),
                            "summary_length": len(summary),
                        },
                    )
                    docs.append(doc)
                    
                except Exception as e:
                    logger.warning(f"Error processing document {i}: {e}")
                    continue
            
            logger.info(f"Successfully downloaded {len(docs)} documents")
            
            # Log statistics
            if docs:
                avg_text_len = sum(len(d.text) for d in docs) / len(docs)
                avg_summary_len = sum(len(d.summary) for d in docs) / len(docs)
                logger.info(
                    f"Statistics: "
                    f"Avg text length: {avg_text_len:.0f} chars, "
                    f"Avg summary length: {avg_summary_len:.0f} chars"
                )
            
            return docs
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            raise
    
    @classmethod
    def save(
        cls,
        docs: List[Document],
        dataset_name: str,
        split: str = "train",
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save documents to JSONL file.
        
        Args:
            docs: List of Document objects
            dataset_name: Name of dataset
            split: Data split
            output_dir: Output directory (default: RAW_DIR)
        
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = RAW_DIR
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{dataset_name}_{split}.jsonl"
        
        logger.info(f"Saving {len(docs)} documents to {output_path}")
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for doc in tqdm(docs, desc="Saving documents"):
                    f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
            
            logger.info(f"Successfully saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save documents: {e}")
            raise
    
    @classmethod
    def load(
        cls,
        dataset_name: str,
        split: str = "train",
        input_dir: Optional[Path] = None,
    ) -> List[Document]:
        """
        Load documents from JSONL file.
        
        Args:
            dataset_name: Name of dataset
            split: Data split
            input_dir: Input directory (default: RAW_DIR)
        
        Returns:
            List of Document objects
        """
        if input_dir is None:
            input_dir = RAW_DIR
        
        input_path = input_dir / f"{dataset_name}_{split}.jsonl"
        
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        logger.info(f"Loading documents from {input_path}")
        
        docs = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading documents"):
                try:
                    data = json.loads(line)
                    docs.append(Document(**data))
                except Exception as e:
                    logger.warning(f"Error loading document: {e}")
                    continue
        
        logger.info(f"Loaded {len(docs)} documents")
        return docs
    
    @classmethod
    def download_and_save(
        cls,
        dataset_name: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
    ) -> Path:
        """
        Download dataset and save to disk (with caching).
        
        Args:
            dataset_name: Name of dataset
            split: Data split
            max_samples: Maximum number of samples
            cache_dir: Cache directory
            force_download: Force re-download even if file exists
        
        Returns:
            Path to saved file
        """
        output_path = RAW_DIR / f"{dataset_name}_{split}.jsonl"
        
        # Check if already downloaded
        if output_path.exists() and not force_download:
            logger.info(f"Dataset already exists at {output_path}")
            logger.info("Use force_download=True to re-download")
            return output_path
        
        # Download and save
        docs = cls.download(
            dataset_name=dataset_name,
            split=split,
            max_samples=max_samples,
            cache_dir=cache_dir,
        )
        return cls.save(docs, dataset_name, split)

