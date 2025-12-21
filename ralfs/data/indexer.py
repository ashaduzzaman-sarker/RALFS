# ============================================================================
# File: ralfs/data/indexer.py
# ============================================================================
"""Build retrieval indexes (FAISS, BM25, ColBERT)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

# Retrieval libraries
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ralfs.core.constants import INDEX_DIR, PROCESSED_DIR
from ralfs.core.logging import get_logger
from ralfs.utils.io import load_jsonl, save_json

logger = get_logger(__name__)


class IndexBuilder:
    """Build and manage retrieval indexes."""
    
    def __init__(self, cfg):
        """
        Initialize index builder.
        
        Args:
            cfg: RALFSConfig object
        """
        self.cfg = cfg
        self.dataset_name = cfg.data.dataset
        self.split = cfg.data.split
        
        # Paths
        self.chunks_path = PROCESSED_DIR / f"{self.dataset_name}_{self.split}_chunks.jsonl"
        self.index_dir = INDEX_DIR / self.dataset_name
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized index builder for {self.dataset_name}")
    
    def load_chunks(self) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Load chunks and extract texts.
        
        Returns:
            Tuple of (texts, chunk_dicts)
        """
        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {self.chunks_path}. "
                "Run preprocessing first."
            )
        
        logger.info(f"Loading chunks from {self.chunks_path}")
        chunks = load_jsonl(self.chunks_path)
        
        texts = [c["text"] for c in chunks]
        logger.info(f"Loaded {len(texts)} chunks")
        
        return texts, chunks
    
    def build_dense_index(self, force_rebuild: bool = False) -> Path:
            """
            Build FAISS dense index.
            
            Args:
                force_rebuild: Force rebuild even if exists
            
            Returns:
                Path to FAISS index
            """
            index_path = self.index_dir / "faiss.index"
            embeddings_path = self.index_dir / "embeddings.npy"
            metadata_path = self.index_dir / "metadata.json"
            
            # Check if already exists
            if index_path.exists() and not force_rebuild:
                logger.info(f"FAISS index already exists at {index_path}")
                logger.info("Use force_rebuild=True to rebuild")
                return index_path
            
            # Load chunks
            texts, chunks = self.load_chunks()
            
            # Load model
            dense_config = getattr(self.cfg.retriever, 'dense', None)
            if dense_config and hasattr(dense_config, 'model'):
                model_name = dense_config.model
            else:
                model_name = getattr(self.cfg.retriever, 'dense_model', 'sentence-transformers/all-MiniLM-L12-v2')
            
            logger.info(f"Loading dense model: {model_name}")
            model = SentenceTransformer(model_name)
            
            # Encode texts
            logger.info(f"Encoding {len(texts)} chunks...")
            embeddings = model.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            embeddings = embeddings.astype('float32')
            
            logger.info(f"Embeddings shape: {embeddings.shape}")
            
            # Build FAISS index
            dim = embeddings.shape[1]
            
            # Get FAISS config
            faiss_config = getattr(self.cfg.retriever, 'faiss', None)
            if faiss_config:
                index_type = getattr(faiss_config, 'index_type', 'IVFFlat')
                nlist = getattr(faiss_config, 'nlist', 100)
                nprobe = getattr(faiss_config, 'nprobe', 10)
                use_gpu = getattr(faiss_config, 'use_gpu', False)
            else:
                index_type = 'IVFFlat'
                nlist = 100
                nprobe = 10
                use_gpu = False
            
            # Adjust nlist for small datasets (FAISS requires n_samples >= nlist)
            n_samples = len(embeddings)
            if index_type in ["IVFFlat", "IVFPQ"] and nlist >= n_samples:
                # Use Flat index for very small datasets or adjust nlist
                if n_samples < 39:  # Minimum practical size for IVF
                    logger.warning(
                        f"Dataset too small ({n_samples} samples) for IVF index, "
                        "using Flat index instead"
                    )
                    index_type = "Flat"
                else:
                    # Adjust nlist to be smaller than n_samples
                    old_nlist = nlist
                    nlist = max(int(np.sqrt(n_samples)), min(nlist, n_samples // 2))
                    logger.warning(
                        f"Adjusted nlist from {old_nlist} to {nlist} "
                        f"for dataset with {n_samples} samples"
                    )
            
            logger.info(f"Building FAISS index (type: {index_type}, dim: {dim})")
            
            if index_type == "Flat":
                # Simple flat index (exact search)
                index = faiss.IndexFlatIP(dim)
            elif index_type == "IVFFlat":
                # IVF with flat quantizer (faster)
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
                # Train index
                logger.info("Training IVF index...")
                index.train(embeddings)
                index.nprobe = nprobe
            elif index_type == "IVFPQ":
                # IVF with product quantization (most compressed)
                quantizer = faiss.IndexFlatIP(dim)
                m = 8  # number of subquantizers
                bits = 8  # bits per subquantizer
                index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, bits)
                logger.info("Training IVFPQ index...")
                index.train(embeddings)
                index.nprobe = nprobe
            else:
                raise ValueError(f"Unknown index type: {index_type}")
            
            # Add embeddings
            logger.info("Adding embeddings to index...")
            index.add(embeddings)
            
            # GPU support (if requested and available)
            if use_gpu and faiss.get_num_gpus() > 0:
                logger.info("Moving index to GPU...")
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            
            # Save index
            logger.info(f"Saving index to {index_path}")
            faiss.write_index(faiss.index_gpu_to_cpu(index) if use_gpu else index, str(index_path))
            
            # Save embeddings
            logger.info(f"Saving embeddings to {embeddings_path}")
            np.save(embeddings_path, embeddings)
            
            # Save metadata
            metadata = {
                "num_chunks": len(texts),
                "embedding_dim": dim,
                "model_name": model_name,
                "index_type": index_type,
                "nlist": nlist if index_type != "Flat" else None,
                "nprobe": nprobe if index_type != "Flat" else None,
            }
            save_json(metadata, metadata_path)
            
            logger.info(f"✅ Dense index built successfully: {index_path}")
            logger.info(f"   - Chunks: {len(texts)}")
            logger.info(f"   - Dim: {dim}")
            logger.info(f"   - Type: {index_type}")
            
            return index_path
    
    def build_sparse_index(self, force_rebuild: bool = False) -> Path:
        """
        Build BM25 sparse index.
        
        Args:
            force_rebuild: Force rebuild even if exists
        
        Returns:
            Path to BM25 index
        """
        index_path = self.index_dir / "bm25_index.pkl"
        
        # Check if already exists
        if index_path.exists() and not force_rebuild:
            logger.info(f"BM25 index already exists at {index_path}")
            logger.info("Use force_rebuild=True to rebuild")
            return index_path
        
        # Load chunks
        texts, chunks = self.load_chunks()
        
        # Tokenize
        logger.info("Tokenizing texts for BM25...")
        tokenized_corpus = [text.lower().split() for text in tqdm(texts, desc="Tokenizing")]
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        
        # Get BM25 hyperparameters
        sparse_config = getattr(self.cfg.retriever, 'sparse', None)
        if sparse_config:
            k1 = getattr(sparse_config, 'bm25_k1', 1.5)
            b = getattr(sparse_config, 'bm25_b', 0.75)
        else:
            k1 = 1.5
            b = 0.75
        
        bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        
        # Save index
        logger.info(f"Saving BM25 index to {index_path}")
        with open(index_path, 'wb') as f:
            pickle.dump(bm25, f)
        
        logger.info(f"✅ BM25 index built successfully: {index_path}")
        logger.info(f"   - Chunks: {len(texts)}")
        logger.info(f"   - k1: {k1}, b: {b}")
        
        return index_path
    
    def build_all_indexes(self, force_rebuild: bool = False) -> Dict[str, Path]:
        """
        Build all indexes (dense + sparse).
        
        Args:
            force_rebuild: Force rebuild even if exists
        
        Returns:
            Dict mapping index type to path
        """
        logger.info("Building all indexes...")
        
        indexes = {}
        
        # Dense index (FAISS)
        try:
            indexes['dense'] = self.build_dense_index(force_rebuild=force_rebuild)
        except Exception as e:
            logger.error(f"Failed to build dense index: {e}")
        
        # Sparse index (BM25)
        try:
            indexes['sparse'] = self.build_sparse_index(force_rebuild=force_rebuild)
        except Exception as e:
            logger.error(f"Failed to build sparse index: {e}")
        
        logger.info(f"✅ All indexes built: {list(indexes.keys())}")
        return indexes


def build_index(cfg, force_rebuild: bool = False) -> Dict[str, Path]:
    """
    Build retrieval indexes (backward compatible).
    
    Args:
        cfg: RALFSConfig object
        force_rebuild: Force rebuild even if exists
    
    Returns:
        Dict mapping index type to path
    """
    builder = IndexBuilder(cfg)
    return builder.build_all_indexes(force_rebuild=force_rebuild)


def build_sparse_index(cfg, force_rebuild: bool = False) -> Path:
    """
    Build sparse (BM25) index only.
    
    Args:
        cfg: RALFSConfig object
        force_rebuild: Force rebuild even if exists
    
    Returns:
        Path to BM25 index
    """
    builder = IndexBuilder(cfg)
    return builder.build_sparse_index(force_rebuild=force_rebuild)