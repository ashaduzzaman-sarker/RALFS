# src/ralfs/data/processor.py
from __future__ import annotations
from typing import List
from pathlib import Path
from ralfs.core.logging import get_logger
from ralfs.core.constants import PROCESSED_DIR, RAW_DIR
from ralfs.data.downloader import DatasetDownloader
from ralfs.data.chunker import SemanticChunker
from ralfs.utils.io import save_json

logger = get_logger(__name__)

def run_preprocessing(cfg) -> None:
    dataset_name = cfg.data.dataset
    max_samples = cfg.data.get("max_samples", None)

    logger.info(f"Starting preprocessing for {dataset_name}")

    # 1. Download
    docs = DatasetDownloader.download_and_save(
        dataset_name=dataset_name,
        split="train",
        max_samples=max_samples
    )

    # 2. Chunk
    chunker = SemanticChunker(
        chunk_size=cfg.data.chunk_size,
        overlap=cfg.data.overlap
    )
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk(doc.text, doc.id)
        for c in chunks:
            c.metadata.update({
                "doc_id": doc.id,
                "title": doc.title,
                "summary": doc.summary,
                "domain": doc.domain
            })
        all_chunks.extend(chunks)

    # 3. Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{dataset_name}_chunks.jsonl"
    save_json([chunk.to_dict() for chunk in all_chunks], out_path, as_jsonl=True)

    logger.info(f"Saved {len(all_chunks)} chunks → {out_path}")