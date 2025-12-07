# src/ralfs/generator/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from omegaconf import DictConfig

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, passages: List[Dict]) -> Tuple[str, Dict]:
        pass