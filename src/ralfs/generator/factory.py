# src/ralfs/generator/factory.py
from omegaconf import DictConfig
from .fid import FiDGenerator

def create_generator(cfg: DictConfig) -> FiDGenerator:
    return FiDGenerator(cfg)