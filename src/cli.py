"""RALFS CLI Entry Point."""
from __future__ import annotations

import hydra
from omegaconf import DictConfig
from src.utils.logging import get_logger


@hydra.main(config_path="../configs/train", config_name="fid_base", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """RALFS CLI entrypoint.
    
    Handles all RALFS commands via Hydra configuration.
    
    Args:
        cfg: Hydra-loaded DictConfig containing task and parameters
    """
    logger = get_logger(cfg=cfg)
    logger.info(f"🚀 Starting RALFS with task: {cfg.task}")
    
    try:
        if cfg.task == "preprocess":
            from src.data.preprocess import preprocess_data
            preprocess_data(cfg)
        elif cfg.task == "build_index":
            from src.data.build_index import build_faiss_index
            build_faiss_index(cfg)
        elif cfg.task == "train_retriever":
            from src.retriever.train_retriever import train_retriever
            train_retriever(cfg)
        elif cfg.task == "train_generator":
            from src.generator.train_generator import train_generator
            train_generator(cfg)
        else:
            logger.error(f"Unknown task: {cfg.task}")
            raise ValueError(f"Available tasks: preprocess, build_index, train_retriever, train_generator")
            
    except Exception as e:
        logger.error(f"Task '{cfg.task}' failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()