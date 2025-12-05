# src/ralfs/data/chunker.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import nltk
import re
from ralfs.core.logging import get_logger

logger = get_logger(__name__)
nltk.download('punkt', quiet=True)

@dataclass Chunk:
    text: str
    chunk_id: str
    doc_id: str
    start_char: int
    end_char: int
    metadata: dict

    def __init__(self, text: str, chunk_id: str, doc_id: str, start: int, end: int, metadata: dict):
        self.text = text
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.start_char = start
        self.end_char = end
        self.metadata = metadata

class SemanticChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = nltk.sent_tokenize

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        text = re.sub(r'\s+', ' ', text.strip())
        sentences = self.tokenizer(text)
        chunks = []
        current = []
        current_tokens = 0
        start_pos = 0

        for sent in sentences:
            sent_tokens = len(sent.split())
            if current_tokens + sent_tokens > self.chunk_size and len(current) > 2:
                chunk_text = " ".join(current)
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=f"{doc_id}_c{len(chunks)}",
                    doc_id=doc_id,
                    start_char=start_pos,
                    end_char=start_pos + len(chunk_text),
                    metadata={}
                ))
                # Overlap: keep last 2 sentences
                overlap_text = " ".join(current[-2:])
                current = current[-2:]
                current_tokens = len(overlap_text.split())
                start_pos += len(chunk_text) - len(overlap_text) + 1
            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunk_text = " ".join(current)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"{doc_id}_c{len(chunks)}",
                doc_id=doc_id,
                start_char=start_pos,
                end_char=len(text),
                metadata={}
            ))
        return chunks