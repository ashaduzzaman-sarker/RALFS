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
        logger.info(f"Loaded {len(self.chunks)} chunks")

        self.retriever = HybridRetriever("data/index/faiss.index", self.chunks)
        logger.info("HybridRetriever initialized")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["article"][:1000]  # Use first 1000 chars
        summary = item["abstract"]

        try:
            passages = self.retriever.retrieve(query, k_final=20)
            # Ensure we always return list of dicts with "text"
            if not passages or not isinstance(passages[0], dict):
                passages = [{"text": "No relevant passage found.", "score": 0.0}] * 20
        except Exception as e:
            logger.warning(f"Retrieval failed for idx {idx}: {e}")
            passages = [{"text": "Retrieval error.", "score": 0.0}] * 20

        return query, summary, passages
