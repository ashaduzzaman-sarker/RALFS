# src/ralfs/retriever/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict
from pathlib import Path

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 20) -> List[Dict[str, any]]:
        """Return list of {'text': str, 'score': float, 'doc_id': str}"""
        pass

    @abstractmethod
    def load_index(self, index_path: Path) -> None:
        pass