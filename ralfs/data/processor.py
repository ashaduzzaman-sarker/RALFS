# ============================================================================
# File: ralfs/data/processor.py
# ============================================================================
"""Document processing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from ralfs.core.constants import PROCESSED_DIR, RAW_DIR
from ralfs.core.logging import get_logger
from ralfs.data.chunker import Chunk, create_chunker
from ralfs.data.downloader import DatasetDownloader, Document
from ralfs.utils.io import load_jsonl, save_jsonl

logger = get_logger(__name__)


class DocumentProcessor:
    """Process documents: download → chunk → save."""
    
    def __init__(self, cfg):
        """
        Initialize processor with configuration.
        
        Args:
            cfg: RALFSConfig object
        """
        self.cfg = cfg
        self.dataset_name = cfg.data.dataset
        self.split = cfg.data.split
        self.max_samples = getattr(cfg.data, 'max_samples', None)
        
        # Create chunker based on strategy
        self.chunker = create_chunker(
            strategy=getattr(cfg.data, 'chunking_strategy', 'semantic'),
            chunk_size=cfg.data.chunk_size,
            overlap=cfg.data.overlap,
            min_chunk_size=getattr(cfg.data, 'min_chunk_size', 100),
        )
        
        logger.info(
            f"Initialized processor for {self.dataset_name} "
            f"({self.chunker.__class__.__name__})"
        )
    
    def download_documents(self, force_download: bool = False) -> List[Document]:
        """
        Download documents from HuggingFace.
        
        Args:
            force_download: Force re-download even if cached
        
        Returns:
            List of Document objects
        """
        cache_dir = getattr(self.cfg.data, 'cache_dir', None)
        
        try:
            # Check if already downloaded
            raw_path = RAW_DIR / f"{self.dataset_name}_{self.split}.jsonl"
            if raw_path.exists() and not force_download:
                logger.info(f"Loading cached documents from {raw_path}")
                return DatasetDownloader.load(self.dataset_name, self.split)
            
            # Download fresh
            docs = DatasetDownloader.download(
                dataset_name=self.dataset_name,
                split=self.split,
                max_samples=self.max_samples,
                cache_dir=cache_dir,
            )
            
            # Save for caching
            DatasetDownloader.save(docs, self.dataset_name, self.split)
            
            return docs
            
        except Exception as e:
            logger.error(f"Failed to download documents: {e}")
            raise
    
    def chunk_documents(self, docs: List[Document]) -> List[Chunk]:
        """
        Chunk documents into smaller pieces.
        
        Args:
            docs: List of Document objects
        
        Returns:
            List of Chunk objects
        """
        logger.info(f"Chunking {len(docs)} documents...")
        
        all_chunks = []
        failed_docs = 0
        
        for doc in tqdm(docs, desc="Chunking documents"):
            try:
                # Chunk document
                chunks = self.chunker.chunk(doc.text, doc.id)
                
                # Add document metadata to each chunk
                for chunk in chunks:
                    chunk.metadata.update({
                        "doc_id": doc.id,
                        "title": doc.title,
                        "summary": doc.summary,
                        "domain": doc.domain,
                        "source": doc.source,
                    })
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.warning(f"Failed to chunk document {doc.id}: {e}")
                failed_docs += 1
                continue
        
        if failed_docs > 0:
            logger.warning(f"Failed to chunk {failed_docs} documents")
        
        logger.info(
            f"Created {len(all_chunks)} chunks from {len(docs)} documents "
            f"(avg: {len(all_chunks)/len(docs):.1f} chunks/doc)"
        )
        
        return all_chunks
    
    def save_chunks(self, chunks: List[Chunk]) -> Path:
        """
        Save chunks to JSONL file.
        
        Args:
            chunks: List of Chunk objects
        
        Returns:
            Path to saved file
        """
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PROCESSED_DIR / f"{self.dataset_name}_{self.split}_chunks.jsonl"
        
        logger.info(f"Saving {len(chunks)} chunks to {output_path}")
        
        try:
            chunk_dicts = [chunk.to_dict() for chunk in chunks]
            save_jsonl(chunk_dicts, output_path)
            
            logger.info(f"Successfully saved to {output_path}")
            
            # Log statistics
            avg_chunk_len = sum(len(c.text) for c in chunks) / len(chunks)
            logger.info(
                f"Statistics: "
                f"Total chunks: {len(chunks)}, "
                f"Avg length: {avg_chunk_len:.0f} chars"
            )
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save chunks: {e}")
            raise
    
    def load_chunks(self) -> List[Chunk]:
        """
        Load chunks from disk.
        
        Returns:
            List of Chunk objects
        """
        input_path = PROCESSED_DIR / f"{self.dataset_name}_{self.split}_chunks.jsonl"
        
        if not input_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {input_path}")
        
        logger.info(f"Loading chunks from {input_path}")
        
        chunk_dicts = load_jsonl(input_path)
        chunks = [Chunk(**d) for d in chunk_dicts]
        
        logger.info(f"Loaded {len(chunks)} chunks")
        return chunks
    
    def process(self, force_download: bool = False, force_rechunk: bool = False) -> Path:
        """
        Full processing pipeline: download → chunk → save.
        
        Args:
            force_download: Force re-download documents
            force_rechunk: Force re-chunk even if cached
        
        Returns:
            Path to saved chunks file
        """
        logger.info(f"Starting full preprocessing pipeline for {self.dataset_name}")
        
        # Check if chunks already exist
        chunks_path = PROCESSED_DIR / f"{self.dataset_name}_{self.split}_chunks.jsonl"
        if chunks_path.exists() and not force_rechunk and not force_download:
            logger.info(f"Chunks already exist at {chunks_path}")
            logger.info("Use force_rechunk=True to reprocess")
            return chunks_path
        
        # Download documents
        docs = self.download_documents(force_download=force_download)
        
        # Chunk documents
        chunks = self.chunk_documents(docs)
        
        # Save chunks
        output_path = self.save_chunks(chunks)
        
        logger.info(f"Preprocessing complete! Output: {output_path}")
        return output_path


def run_preprocessing(cfg, force_download: bool = False, force_rechunk: bool = False) -> Path:
    """
    Main preprocessing function (backward compatible).
    
    Args:
        cfg: RALFSConfig object
        force_download: Force re-download documents
        force_rechunk: Force re-chunk even if cached
    
    Returns:
        Path to saved chunks file
    """
    processor = DocumentProcessor(cfg)
    return processor.process(
        force_download=force_download,
        force_rechunk=force_rechunk,
    )
