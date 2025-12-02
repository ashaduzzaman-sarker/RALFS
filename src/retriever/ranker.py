"""Cross-Encoder Reranker."""
from sentence_transformers import CrossEncoder
from src.utils.logging import get_logger
from pathlib import Path
from typing import List, Dict, Tuple

logger = get_logger(__name__)

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model = CrossEncoder(model_name, max_length=512)
        logger.info(f"Loaded CrossEncoder: {model_name}")

    def rerank(self, query: str, documents: List[str], top_k: int = 20) -> List[Dict]:
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs, batch_size=32)
        ranked = sorted(zip(documents, scores), key=lambda x: -x[1])[:top_k]
        return [
            {"text": doc, "score": float(score), "rank": i + 1}
            for i, (doc, score) in enumerate(ranked)
        ]