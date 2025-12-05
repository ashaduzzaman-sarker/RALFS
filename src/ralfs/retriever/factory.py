# src/ralfs/retriever/factory.py
from omegaconf import DictConfig
from .hybrid import HybridRetriever

def create_retriever(cfg: DictConfig) -> HybridRetriever:
    return HybridRetriever(cfg)