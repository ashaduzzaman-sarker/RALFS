# src/training/fid_dataset.py
from datasets import load_dataset
from torch.utils.data import Dataset
from src.retriever.hybrid import HybridRetriever
from src.utils.io import RALFSDataManager

class FiDDataset(Dataset):
    def __init__(self, split="train"):
        self.data = load_dataset("ccdv/arxiv-summarization", split=split)
        # Load chunks exactly like in Phase 2
        metadata = RALFSDataManager.load_json("data/index/faiss.metadata.json")
        self.chunks = metadata["texts"]
        self.retriever = HybridRetriever("data/index/faiss.index", self.chunks)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["report_title"]
        summary = item["summary"]
        passages = self.retriever.retrieve(query, k_final=20)
        return query, summary, passages
