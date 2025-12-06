# examples/retriever_demo.py
import hydra
from omegaconf import DictConfig
from ralfs.retriever.factory import create_retriever
from ralfs.core.logging import get_logger


log = get_logger()

@hydra.main(config_path="../configs", config_name="ralfs", version_base="1.3")
def main(cfg: DictConfig):

    retriever = create_retriever(cfg)

    queries = [
        "attention is all you need",
        "long document summarization",
        "hybrid retrieval with colbert",
        "cross encoder reranking"
    ]

    for q in queries:
        log.info(f"\nQuery: {q}")
        results = retriever.retrieve(q)
        for r in results[:3]:
            print(f"{r['rank']}. {r['score']:.4f} | {r['text'][:100]}...")

if __name__ == "__main__":
    main()
