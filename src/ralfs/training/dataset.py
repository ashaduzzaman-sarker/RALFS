# src/ralfs/training/dataset.py
from __future__ import annotations
from torch.utils.data import Dataset
from ralfs.retriever.factory import create_retriever
from ralfs.utils.io import load_json
from ralfs.core.logging import get_logger

logger = get_logger(__name__)

class FiDDataset(Dataset):
    def __init__(self, cfg, split: str = "train"):
        self.cfg = cfg
        self.retriever = create_retriever(cfg)
        
        # Load preprocessed chunks — this is what preprocessing saved
        chunks_path = f"data/processed/{cfg.data.dataset}_chunks.jsonl"
        self.chunks = load_json(chunks_path, as_jsonl=True)
        
        # Limit for debugging
        max_samples = cfg.data.get("max_samples", len(self.chunks))
        self.chunks = self.chunks[:max_samples]
        
        logger.info(f"FiDDataset loaded {len(self.chunks)} chunks for training")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        query = chunk["text"][:1000]  # Use chunk as query (self-supervised)
        
        # Retrieve relevant passages
        passages = self.retriever.retrieve(query)
        
        # Target is the document's gold summary (from metadata)
        target_summary = chunk["metadata"].get("summary", "No summary available")
        
        # FiD input format
        inputs = [f"question: {query} context: {p['text']}" for p in passages[:20]]
        
        return {
            "inputs": inputs,
            "summaries": [target_summary]
        }
