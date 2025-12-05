# src/ralfs/retriever/utils.py
from typing import List, Dict

def reciprocal_rank_fusion(results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
    """Standard RRF fusion"""
    fused = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            text = doc["text"]
            fused[text] = fused.get(text, 0) + 1 / (k + rank + 1)
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return [{"text": t, "score": s, "rank": i+1} for i, (t, s) in enumerate(ranked)]