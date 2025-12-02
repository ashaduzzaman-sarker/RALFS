from src.retriever.hybrid import HybridRetriever
from src.retriever.ranker import CrossEncoderReranker
from src.retriever.adaptive_fid import AdaptiveFiD
from src.utils.io import RALFSDataManager
from pathlib import Path

# Load chunks
metadata = RALFSDataManager.load_json("data/index/faiss.metadata.json")
chunks = metadata["texts"]

# Initialize systems
hybrid = HybridRetriever("data/index/faiss.index", chunks)
reranker = CrossEncoderReranker()
generator = AdaptiveFiD()

query = "What are the main findings of the GAO report on Medicare Advantage?"

print("=== Phase 2: Full Hybrid + Rerank + AdaptiveFiD ===\n")
hybrid_results = hybrid.retrieve(query, k_final=20)
print(f"Hybrid retrieved top-20 (fused dense+bm25+colbert)")

reranked = reranker.rerank(query, [r["text"] for r in hybrid_results[:20]])
print(f"Cross-encoder reranked → top-5")

summary, k_used = generator.generate(query, reranked)
print(f"\nAdaptiveFiD used k={k_used} passages")
print(f"Generated Summary:\n{summary}")