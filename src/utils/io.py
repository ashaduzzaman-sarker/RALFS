"""src/utils/io.py
RALFS I/O Utilities."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np


class RALFSDataManager:
    """Unified I/O manager for RALFS data artifacts."""
    
    @staticmethod
    def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
        """Save data as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    
    @staticmethod
    def load_json(path: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON data."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_pickle(data: Any, path: Union[str, Path]) -> None:
        """Save data using pickle (for embeddings/indexes)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_pickle(path: Union[str, Path]) -> Any:
        """Load pickled data."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_embeddings(embeddings: np.ndarray, path: Union[str, Path]) -> None:
        """Save FAISS-compatible embeddings."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
    
    @staticmethod
    def load_embeddings(path: Union[str, Path]) -> np.ndarray:
        """Load embeddings as numpy array."""
        return np.load(path)
