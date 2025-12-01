from src.retriever.encoder import RALFSEncoder
from src.utils.io import RALFSDataManager
from pathlib import Path
import faiss

def build_faiss_index(cfg):
    logger = get_logger(cfg=cfg)
    logger.info("Building FAISS index")
    
    chunks_path = Path(cfg.data.processed_path) / f"{cfg.data.dataset}_chunks.json"
    chunks = RALFSDataManager.load_json(chunks_path)
    texts = [c["text"] for c in chunks]
    
    encoder = RALFSEncoder(cfg.retriever.model)
    embeddings = encoder.encode(texts, batch_size=32)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))
    
    index_path = Path(cfg.retriever.index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    
    RALFSDataManager.save_json({
        "texts": texts,
        "embedding_dim": dim
    }, index_path.with_suffix('.metadata.json'))
    
    logger.info(f"FAISS Index saved: {index_path} ({index.ntotal} passages)")