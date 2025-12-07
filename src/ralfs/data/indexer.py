# src/ralfs/data/indexer.py
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ralfs.core.logging import get_logger
from ralfs.utils.io import save_json, load_json
from ralfs.core.constants import INDEX_DIR, PROCESSED_DIR

logger = get_logger(__name__)

def build_index(cfg) -> None:
    chunks_path = Path(cfg.retriever.chunks_path)
    chunks = load_json(chunks_path, as_jsonl=True)
    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(cfg.retriever.dense.model)
    logger.info(f"Encoding {len(texts)} chunks with {cfg.retriever.dense.model}")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEX_DIR / "faiss.index"
    faiss.write_index(index, str(index_path))

    save_json({"texts": texts}, index_path.with_suffix(".metadata.json"))
    logger.info(f"FAISS index saved: {index_path}")