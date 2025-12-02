"""Utility functions for hybrid retrieval."""
import numpy as np
from typing import List, Dict

def reciprocal_rank_fusion(results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
    fused_scores = {}
    for docs in results_list:
        for rank, doc in enumerate(docs):
            doc_text = doc["text"]
            if doc_text not in fused_scores:
                fused_scores[doc_text] = 0
            fused_scores[doc_text] += 1 / (k + rank + 1)
    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"text": text, "score": score} for text, score in reranked]