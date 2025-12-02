import pytest
from src.retriever.hybrid import HybridRetriever
from src.retriever.ranker import CrossEncoderReranker

def test_hybrid_retrieval():
    chunks = ["Medicare Advantage is growing.", "GAO found issues in CMS oversight."]
    retriever = HybridRetriever("data/index/faiss.index", chunks)
    results = retriever.retrieve("What did GAO find?", k_final=2)
    assert len(results) == 2
    assert results[0]["score"] > results[1]["score"]

def test_reranker():
    reranker = CrossEncoderReranker()
    docs = ["Good match", "Bad match"]
    results = reranker.rerank("test query", docs)
    assert len(results) == 2