# ============================================================================
# File: ralfs/training/dataset.py 
# ============================================================================
"""Dataset for FiD training with retrieval."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

from ralfs.core.logging import get_logger
from ralfs.utils.io import load_jsonl

logger = get_logger(__name__)


class FiDDataset(Dataset):
    """
    Dataset for Fusion-in-Decoder training.
    
    Each sample contains:
    - query: Input query/question
    - passages: List of retrieved passage texts
    - summary: Target summary
    """
    
    def __init__(
        self,
        data_path: Path | str,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 512,
        max_output_length: int = 200,
        max_passages: int = 20,
        use_retrieved: bool = False,
        retriever = None,
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSONL file with samples
            tokenizer: Tokenizer for encoding
            max_input_length: Max tokens per passage
            max_output_length: Max tokens in summary
            max_passages: Max number of passages to use
            use_retrieved: If True, retrieve passages; if False, use pre-retrieved
            retriever: Retriever instance (only if use_retrieved=True)
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.max_passages = max_passages
        self.use_retrieved = use_retrieved
        self.retriever = retriever
        
        # Load data
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.samples = load_jsonl(self.data_path)
        logger.info(f"Loaded {len(self.samples)} samples from {self.data_path}")
        
        # Validate retriever if needed
        if self.use_retrieved and self.retriever is None:
            raise ValueError("Retriever required when use_retrieved=True")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single training sample.
        
        Returns:
            Dict with 'input_ids', 'attention_mask', 'labels'
        """
        sample = self.samples[idx]
        
        # Extract query and summary
        query = sample.get("text", "")[:1000]  # Use chunk text as query
        summary = sample.get("metadata", {}).get("summary", "")
        
        # Get passages
        if self.use_retrieved:
            # Retrieve on-the-fly
            results = self.retriever.retrieve(query, k=self.max_passages)
            passages = [r.text for r in results]
        else:
            # Use pre-retrieved passages (if available)
            passages = sample.get("passages", [query])[:self.max_passages]
        
        # Format inputs for FiD
        fid_inputs = [
            f"question: {query} context: {passage}"
            for passage in passages
        ]
        
        # Tokenize inputs
        encoded_inputs = self.tokenizer(
            fid_inputs,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Tokenize labels
        encoded_labels = self.tokenizer(
            summary,
            max_length=self.max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Flatten to batch dimension
        return {
            'input_ids': encoded_inputs['input_ids'].squeeze(0),
            'attention_mask': encoded_inputs['attention_mask'].squeeze(0),
            'labels': encoded_labels['input_ids'].squeeze(0),
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create DataLoader for training.
    
    Args:
        dataset: FiDDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers for loading
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
