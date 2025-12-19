# ============================================================================
# File: ralfs/retriever/sparse.py
# ============================================================================
"""Sparse retrieval using BM25."""

from __future__ import annotations
from typing import List, Optional
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi

from ralfs.retriever.base import BaseRetriever, RetrievalResult
from ralfs.core.logging import get_logger
from ralfs.utils.io import load_jsonl
from ralfs.core.constants import PROCESSED_DIR

logger = get_logger(__name__)


class SparseRetriever(BaseRetriever):
    """
    Sparse retrieval using BM25.
    
    Features:
    - BM25 scoring (industry standard)
    - Fast retrieval for exact matches
    - Configurable BM25 hyperparameters (k1, b)
    """
    
    def __init__(self, cfg, chunks: Optional[List[str]] = None):
        """
        Initialize sparse retriever.
        
        Args:
            cfg: Configuration object
            chunks: Optional pre-loaded chunk texts
        """
        super().__init__(cfg)
        
        # Get config
        sparse_config = getattr(cfg.retriever, 'sparse', None)
        if sparse_config:
            self.k_default = getattr(sparse_config, 'k', 100)
            self.k1 = getattr(sparse_config, 'bm25_k1', 1.5)
            self.b = getattr(sparse_config, 'bm25_b', 0.75)
        else:
            self.k_default = getattr(cfg.retriever, 'k_sparse', 100)
            self.k1 = 1.5
            self.b = 0.75
        
        # BM25 index
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: Optional[List[Dict]] = None
        self.texts: Optional[List[str]] = None
        
        # Build index if chunks provided
        if chunks:
            self._build_index(chunks)
            self._initialized = True
        
        logger.info(f"Sparse retriever initialized (BM25: k1={self.k1}, b={self.b})")
    
    def _build_index(self, texts: List[str]):
        """Build BM25 index from texts."""
        logger.info(f"Building BM25 index from {len(texts)} texts...")
        
        # Tokenize
        tokenized_corpus = [text.lower().split() for text in texts]
        
        # Build BM25
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self.texts = texts
        
        logger.info("BM25 index built successfully")
    
    def load_index(self, index_path: Optional[Path] = None) -> None:
        """
        Load chunks and build BM25 index.
        
        Note: BM25 doesn't need pre-built index, builds on-the-fly from chunks.
        
        Args:
            index_path: Not used (for API compatibility)
        """
        # Load chunks
        chunks_path = getattr(self.cfg.retriever, 'chunks_path', None)
        if chunks_path:
            chunks_path = Path(chunks_path)
        else:
            # Default path
            dataset = self.cfg.data.dataset
            split = getattr(self.cfg.data, 'split', 'train')
            chunks_path = PROCESSED_DIR / f"{dataset}_{split}_chunks.jsonl"
        
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {chunks_path}. "
                "Run 'ralfs preprocess' first."
            )
        
        logger.info(f"Loading chunks from {chunks_path}")
        self.chunks = load_jsonl(chunks_path)
        texts = [c["text"] for c in self.chunks]
        
        # Build BM25 index
        self._build_index(texts)
        
        self._initialized = True
        logger.info(f"Sparse retriever loaded with {len(texts)} documents")
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using BM25.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of RetrievalResult objects
        """
        self._ensure_initialized()
        
        if k is None:
            k = self.k_default
        
        # Limit k to corpus size
        k = min(k, len(self.texts))
        
        try:
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:k]
            
            # Convert to results
            results = []
            for rank, idx in enumerate(top_indices, 1):
                score = scores[idx]
                text = self.texts[idx]
                
                # Get chunk metadata
                chunk = self.chunks[idx] if self.chunks else {}
                doc_id = chunk.get("doc_id")
                chunk_id = chunk.get("chunk_id")
                metadata = {
                    "retriever": "sparse",
                    "index": int(idx),
                    **chunk.get("metadata", {}),
                }
                
                results.append(RetrievalResult(
                    text=text,
                    score=float(score),
                    rank=rank,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    metadata=metadata,
                ))
            
            logger.debug(f"Sparse retrieval: {len(results)} results for query '{query[:50]}'")
            return results
            
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            raise
