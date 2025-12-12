# src/ralfs/cli.py
from __future__ import annotations
import hydra
from omegaconf import DictConfig, OmegaConf
from ralfs.core.logging import get_logger

log = get_logger()

@hydra.main(config_path="../configs", config_name="ralfs", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("RALFS v1.0 — Starting task: [bold green]%s[/]", cfg.task)

    if cfg.task == "preprocess":
        from ralfs.data.processor import run_preprocessing
        run_preprocessing(cfg)

    elif cfg.task == "build_index":
        from ralfs.retriever.indexer import build_index
        build_index(cfg)

    elif cfg.task == "retrieve":
        from ralfs.retriever.factory import create_retriever
        retriever = create_retriever(cfg)
        results = retriever.retrieve(cfg.query)
        print(f"\nTop {len(results)} results for: {cfg.query}\n")
        for r in results[:10]:
            print(f"{r['rank']:2d}. {r['score']:.4f} | {r['text'][:120]}...")

    elif cfg.task == "generate":
        from ralfs.retriever.factory import create_retriever
        from ralfs.generator.factory import create_generator
        
        retriever = create_retriever(cfg)
        generator = create_generator(cfg)
        
        log.info("Retrieving passages...")
        results = retriever.retrieve(cfg.query)
        log.info("Generating summary...")
        summary, stats = generator.generate(cfg.query, results)
        
        print(f"\nQuery: {cfg.query}")
        print(f"Used {stats['k_used']} passages out of {len(results)} retrieved")
        print(f"\nSummary:\n{summary}\n")

    elif cfg.task == "train":
        from ralfs.training.trainer import train
        train(cfg)

    elif cfg.task == "evaluate":
        from ralfs.evaluation.main import evaluate
        evaluate(cfg)

    else:
        log.error("Unknown task: %s", cfg.task)
        log.info("Available tasks: preprocess, build_index, retrieve, generate, train, evaluate")
        raise ValueError(f"Invalid task: {cfg.task}")

if __name__ == "__main__":
    main()
