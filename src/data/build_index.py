"""src/data/build_index.py
FAISS Index Builder for RALFS Retriever."""
from __future__ import annotations

import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from omegaconf import DictConfig
from src.utils.logging import get_logger
from src.utils.io import RALFSDataManager
from src.data.preprocess import SemanticChunker


logger = get_logger(__name__)


def build_faiss_index(cfg: DictConfig) -> None:
    """Build FAISS index from preprocessed chunks."""
    logger = get_logger(cfg=cfg)
    logger.info("🚀 Building FAISS Index")
    
    # Load chunks
    chunks_path = Path(cfg.data.path) / "processed" / "chunks.json"
    chunks = RALFSDataManager.load_json(chunks_path)
    
    # Extract texts
    texts = [chunk["text"] for chunk in chunks]
    chunk_ids = [chunk["id"] for chunk in chunks]
    
    logger.info(f"Encoding {len(texts)} chunks...")
    
    # Load retriever model
    retriever_model = SentenceTransformer(cfg.retriever.model)
    
    # Generate embeddings
    embeddings = retriever_model.encode(
        texts,
        batch_size=32,  # T4-friendly
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
    
    logger.info(f"Training FAISS index with {len(embeddings)} vectors (dim={dimension})")
    index.add(embeddings.astype('float32'))
    
    # Save index and metadata
    index_path = Path(cfg.retriever.index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(index_path))
    RALFSDataManager.save_json(
        {"chunk_ids": chunk_ids, "embedding_dim": dimension},
        index_path.with_suffix('.metadata.json')
    )
    RALFSDataManager.save_embeddings(embeddings, index_path.with_suffix('.embeddings.npy'))
    
    logger.info(f"✅ FAISS Index saved: {index_path}")
    logger.info(f"📊 Index stats: {index.ntotal} vectors, {dimension} dimensions")