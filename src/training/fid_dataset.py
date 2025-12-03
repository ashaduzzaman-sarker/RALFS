# src/training/fid_dataset.py
from datasets import load_dataset
from torch.utils.data import Dataset
from src.retriever.hybrid import HybridRetriever
from src.utils.io import RALFSDataManager
from src.utils.logging import get_logger

logger = get_logger(__name__)

class FiDDataset(Dataset):
    def __init__(self, split="train"):
        self.data = load_dataset("ccdv/arxiv-summarization", split=split)
        logger.info(f"Loaded {len(self.data)} arXiv papers")

        metadata = RALFSDataManager.load_json("data/index/faiss.metadata.json")
        self.chunks = metadata["texts"]
        logger.info(f"Loaded {len(self.chunks)} chunks from FAISS index")

        self.retriever = HybridRetriever("data/index/faiss.index", self.chunks)
        logger.info("HybridRetriever ready (Dense + BM25 + ColBERT)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["article"][:2000]  # First 2000 chars as query
        summary = item["abstract"]
        passages = self.retriever.retrieve(query, k_final=20)
        return query, summary, passages
