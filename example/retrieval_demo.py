from src.retriever.retriever import DenseRetriever
from src.utils.logging import get_logger
from pathlib import Path

logger = get_logger(__name__)

retriever = DenseRetriever(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    index_path="data/index/faiss.index"
)

queries = [
    "What are the main findings of the GAO report on Medicare Advantage?",
    "How much money was recovered from risk adjustment audits?",
    "What oversight issues exist in CMS encounter data?"
]

for q in queries:
    print(f"\nQUERY: {q}")
    results = retriever.retrieve(q, k=5)
    for r in results:
        print(f"{r['rank']}. Score: {r['score']:.3f} | {r['text'][:200]}...")