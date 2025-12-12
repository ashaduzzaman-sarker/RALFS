# src/ralfs/training/dataset.py
from torch.utils.data import Dataset
from ralfs.retriever.factory import create_retriever
from ralfs.utils.io import load_json
from ralfs.core.logging import get_logger

logger = get_logger(__name__)

class FiDDataset(Dataset):
    def __init__(self, cfg, split: str = "train"):
        self.cfg = cfg
        self.retriever = create_retriever(cfg)
        self.data = load_json(f"data/{cfg.data.dataset}/{split}.jsonl", as_jsonl=True)
        logger.info(f"Loaded {len(self.data)} examples for {split}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["query"]
        passages = self.retriever.retrieve(query)
        inputs = [f"question: {query} context: {p['text']}" for p in passages[:20]]
        return {
            "inputs": inputs,
            "summary": item["summary"]
        }
