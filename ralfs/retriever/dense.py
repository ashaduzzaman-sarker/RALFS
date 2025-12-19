# ============================================================================
# File: ralfs/retriever/dense.py
# ============================================================================
"""Dense retrieval using FAISS and sentence transformers."""

from __future__ import annotations
from typing import List, Optional
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from ralfs.retriever.base import BaseRetriever, RetrievalResult
from ralfs.core.logging import get_logger
from ralfs.utils.io import load_json, load_jsonl
from ralfs.core.constants import INDEX_DIR, PROCESSED_DIR

logger = get_logger(__name__)


class DenseRetriever(BaseRetriever):
    """
    Dense retrieval using FAISS and sentence transformers.
    
    Features:
    - FAISS index for fast similarity search
    - Sentence transformer embeddings
    - L2 normalization for inner product
    - GPU support (if available)
    """
    
    def __init__(self, cfg):
        """Initialize dense retriever."""
        super().__init__(cfg)
        
        # Get config
        dense_config = getattr(cfg.retriever, 'dense', None)
        if dense_config and hasattr(dense_config, 'model'):
            model_name = dense_config.model
            self.k_default = getattr(dense_config, 'k', 100)
            self.batch_size = getattr(dense_config, 'batch_size', 32)
            self.normalize = getattr(dense_config, 'normalize', True)
        else:
            model_name = getattr(cfg.retriever, 'dense_model', 'sentence-transformers/all-MiniLM-L12-v2')
            self.k_default = getattr(cfg.retriever, 'k_dense', 100)
            self.batch_size = 32
            self.normalize = True
        
        # Load model
        logger.info(f"Loading dense retrieval model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.eval()
        
        # Index and chunks
        self.index: Optional[faiss.Index] = None
        self.chunks: Optional[List[dict]] = None
        self.texts: Optional[List[str]] = None
        
        logger.info(f"Dense retriever initialized (model: {model_name})")
    
    def load_index(self, index_path: Optional[Path] = None) -> None:
        """
        Load FAISS index and chunk metadata.
        
        Args:
            index_path: Path to FAISS index (None = use default from config)
        """
        # Get index path
        if index_path is None:
            if hasattr(self.cfg.retriever, 'index_path'):
                index_path = Path(self.cfg.retriever.index_path)
            else:
                # Default path
                dataset = self.cfg.data.dataset
                index_path = INDEX_DIR / dataset / "faiss.index"
        else:
            index_path = Path(index_path)
        
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}. "
                "Run 'ralfs build-index' first."
            )
        
        logger.info(f"Loading FAISS index from {index_path}")
        
        try:
            # Load index
            self.index = faiss.read_index(str(index_path))
            logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")
            
            # Load metadata
            metadata_path = index_path.with_suffix(".metadata.json")
            if metadata_path.exists():
                metadata = load_json(metadata_path)
                logger.info(f"Metadata loaded: {metadata.get('model_name', 'unknown')}")
            
            # Determine chunks path
            chunks_path = None
            
            # Try config path first
            if hasattr(self.cfg.retriever, 'chunks_path'):
                chunks_path = Path(self.cfg.retriever.chunks_path)
            
            # If not found, use default path
            if not chunks_path or not chunks_path.exists():
                dataset = self.cfg.data.dataset
                split = getattr(self.cfg.data, 'split', 'train')
                chunks_path = PROCESSED_DIR / f"{dataset}_{split}_chunks.jsonl"
            
            # Load chunks - THIS IS REQUIRED
            if chunks_path and chunks_path.exists():
                logger.info(f"Loading chunks from {chunks_path}")
                self.chunks = load_jsonl(chunks_path)
                self.texts = [c["text"] for c in self.chunks]
                
                # Verify chunk count matches index
                if len(self.texts) != self.index.ntotal:
                    logger.warning(
                        f"Chunk count mismatch: {len(self.texts)} chunks "
                        f"vs {self.index.ntotal} vectors in index"
                    )
            else:
                # Raise error if chunks not found (they are required for retrieval)
                raise FileNotFoundError(
                    f"Chunks file not found: {chunks_path}. "
                    "Run 'ralfs preprocess' first."
                )
            
            self._initialized = True
            logger.info("Dense retriever loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load dense index: {e}")
            raise
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using dense retrieval.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of RetrievalResult objects
        """
        self._ensure_initialized()
        
        if k is None:
            k = self.k_default
        
        # Limit k to index size
        k = min(k, self.index.ntotal)
        
        # Additional safety check
        if self.texts is None:
            raise RuntimeError(
                "Chunks not loaded. This should not happen after initialization. "
                "Please report this bug."
            )
        
        try:
            # Encode query
            query_embedding = self.model.encode(
                [query],
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            query_embedding = query_embedding.astype('float32')
            
            # Search
            scores, indices = self.index.search(query_embedding, k)
            
            # Convert to results
            results = []
            seen_texts = set()
            
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
                # Skip invalid indices
                if idx == -1 or idx >= len(self.texts):
                    continue
                
                text = self.texts[idx]
                
                # Skip duplicates
                if text in seen_texts:
                    continue
                seen_texts.add(text)
                
                # Get chunk metadata
                chunk = self.chunks[idx] if self.chunks else {}
                doc_id = chunk.get("doc_id")
                chunk_id = chunk.get("chunk_id")
                metadata = {
                    "retriever": "dense",
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
            
            logger.debug(f"Dense retrieval: {len(results)} results for query '{query[:50]}'")
            return results
            
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            raise