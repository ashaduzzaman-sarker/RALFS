"""src/data/preprocess.py
RALFS Data Preprocessing with Semantic Chunking."""
from __future__ import annotations

import nltk
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from src.utils.logging import get_logger
from omegaconf import DictConfig
import numpy as np

from pathlib import Path
logger = get_logger(__name__)

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('perluniprops', quiet=True)


@dataclass
class Chunk:
    """Semantic chunk representation."""
    text: str
    chunk_id: str
    start_pos: int
    end_pos: int
    sentence_count: int
    tokens: int
    metadata: Dict[str, Any]


class SemanticChunker:
    """Advanced semantic chunking for long documents."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    
    def chunk_document(self, doc_text: str, doc_id: str) -> List[Chunk]:
        """Split document into semantically coherent chunks.
        
        Strategy:
        1. Sentence-level splitting
        2. Semantic grouping (paragraph-like)
        3. Token limit enforcement with overlap
        """
        # Clean text
        doc_text = self._clean_text(doc_text)
        
        # Split into sentences
        sentences = self.tokenizer.tokenize(doc_text)
        chunks: List[Chunk] = []
        
        current_chunk = []
        current_tokens = 0
        start_pos = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = len(sentence.split())
            
            # Start new chunk if adding sentence exceeds limit
            if (current_tokens + sentence_tokens > self.chunk_size and 
                len(current_chunk) > 2):  # Minimum 3 sentences per chunk
                
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                    start_pos=start_pos,
                    end_pos=start_pos + len(chunk_text),
                    sentence_count=len(current_chunk),
                    tokens=current_tokens,
                    metadata={"doc_id": doc_id}
                )
                chunks.append(chunk)
                
                # Slide window with overlap
                start_pos += max(0, current_tokens - self.overlap)
                current_chunk = current_chunk[-2:]  # Keep last 2 sentences
                current_tokens = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk = Chunk(
                text=chunk_text,
                chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                start_pos=start_pos,
                end_pos=len(doc_text),
                sentence_count=len(current_chunk),
                tokens=current_tokens,
                metadata={"doc_id": doc_id}
            )
            chunks.append(chunk)
        
        logger.info(f"Chunked document {doc_id}: {len(chunks)} chunks")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


def preprocess_data(cfg: DictConfig) -> None:
    """Main preprocessing pipeline."""
    from src.data.download import DatasetDownloader
    from src.utils.io import RALFSDataManager
    
    logger = get_logger(cfg=cfg)
    logger.info("🚀 Starting RALFS Data Preprocessing")
    
    # Download dataset
    data = DatasetDownloader.download(
        dataset_name=cfg.data.dataset,
        split="train",
        max_samples=cfg.data.max_samples
    )
    
    # Chunk documents
    chunker = SemanticChunker(
        chunk_size=cfg.data.chunk_size,
        overlap=cfg.data.overlap
    )
    
    all_chunks = []
    for doc in data["documents"]:
        chunks = chunker.chunk_document(doc["text"], doc["id"])
        for chunk in chunks:
            chunk.metadata.update({
                "summary": doc["summary"],
                "title": doc["title"]
            })
        all_chunks.extend(chunks)
    
    # Save processed data
    output_dir = Path(cfg.data.path) / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save chunks
    chunks_data = [{"id": c.chunk_id, "text": c.text, **c.metadata} 
                  for c in all_chunks]
    RALFSDataManager.save_json(chunks_data, output_dir / "chunks.json")
    
    logger.info(f"✅ Preprocessing complete: {len(all_chunks)} chunks saved")