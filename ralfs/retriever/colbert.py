# ============================================================================
# File: ralfs/retriever/colbert.py
# ============================================================================
"""ColBERT late-interaction retrieval."""

from __future__ import annotations
from typing import List, Optional
from pathlib import Path
import warnings

from ralfs.retriever.base import BaseRetriever, RetrievalResult
from ralfs.core.logging import get_logger
from ralfs.utils.io import load_jsonl
from ralfs.core.constants import INDEX_DIR, PROCESSED_DIR

logger = get_logger(__name__)

# Suppress ColBERT warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='colbert')


class ColBERTRetriever(BaseRetriever):
    """
    ColBERT late-interaction retrieval.
    
    Features:
    - Late interaction (MaxSim)
    - State-of-the-art neural retrieval
    - Efficient indexing with compression
    
    Note: Requires ColBERT library installation:
    pip install git+https://github.com/stanford-nlp/ColBERT.git
    """
    
    def __init__(self, cfg):
        """Initialize ColBERT retriever."""
        super().__init__(cfg)
        
        # Get config
        colbert_config = getattr(cfg.retriever, 'colbert', None)
        if colbert_config:
            self.model_name = getattr(colbert_config, 'model', 'colbert-ir/colbertv2.0')
            self.k_default = getattr(colbert_config, 'k', 100)
            self.checkpoint = getattr(colbert_config, 'checkpoint', None)
            self.use_gpu = getattr(colbert_config, 'use_gpu', True)
        else:
            self.model_name = getattr(cfg.retriever, 'colbert_model', 'colbert-ir/colbertv2.0')
            self.k_default = getattr(cfg.retriever, 'k_colbert', 100)
            self.checkpoint = None
            self.use_gpu = True
        
        # ColBERT components
        self.searcher = None
        self.collection_path = None
        self.index_name = "ralfs.colbert"
        
        # Chunks
        self.chunks: Optional[List[Dict]] = None
        self.texts: Optional[List[str]] = None
        
        logger.info(f"ColBERT retriever initialized (model: {self.model_name})")
    
    def _create_collection(self, texts: List[str], output_path: Path):
        """Create ColBERT collection TSV file."""
        logger.info(f"Creating ColBERT collection at {output_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(texts):
                # Clean text (remove tabs and newlines)
                text = text.replace("\t", " ").replace("\n", " ")
                f.write(f"{i}\t{text}\n")
        
        logger.info(f"Collection created with {len(texts)} passages")
    
    def load_index(self, index_path: Optional[Path] = None) -> None:
        """
        Load or build ColBERT index.
        
        Args:
            index_path: Path to index directory (None = use default)
        """
        try:
            # Import ColBERT (optional dependency)
            from colbert import Searcher
            from colbert.infra import Run, RunConfig, ColBERTConfig
            from colbert.data import Collection
            from colbert import Indexer
        except ImportError:
            raise ImportError(
                "ColBERT not installed. Install with: "
                "pip install git+https://github.com/stanford-nlp/ColBERT.git"
            )
        
        # Load chunks
        chunks_path = getattr(self.cfg.retriever, 'chunks_path', None)
        if chunks_path:
            chunks_path = Path(chunks_path)
        else:
            dataset = self.cfg.data.dataset
            split = getattr(self.cfg.data, 'split', 'train')
            chunks_path = PROCESSED_DIR / f"{dataset}_{split}_chunks.jsonl"
        
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {chunks_path}. "
                "Run 'ralfs preprocess' first."
            )
        
        logger.info(f"Loading chunks from {chunks_path}")
        self.chunks = load_jsonl(chunks_path)
        self.texts = [c["text"] for c in self.chunks]
        
        # Create collection
        dataset = self.cfg.data.dataset
        collection_dir = INDEX_DIR / dataset / "colbert"
        collection_dir.mkdir(parents=True, exist_ok=True)
        self.collection_path = collection_dir / "collection.tsv"
        
        if not self.collection_path.exists():
            self._create_collection(self.texts, self.collection_path)
        
        # Index path
        if index_path is None:
            index_dir = INDEX_DIR / dataset / "colbert" / "indexes"
        else:
            index_dir = Path(index_path).parent
        
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if index exists
        full_index_path = index_dir / self.index_name
        
        if not full_index_path.exists():
            logger.info("Building ColBERT index (this may take a while)...")
            
            # Build index
            with Run().context(RunConfig(nranks=1, experiment=str(index_dir.parent.name))):
                config = ColBERTConfig(
                    doc_maxlen=300,
                    nbits=2,
                    kmeans_niters=4,
                    checkpoint=self.checkpoint or self.model_name,
                )
                
                indexer = Indexer(
                    checkpoint=self.checkpoint or self.model_name,
                    config=config,
                )
                
                collection = Collection(path=str(self.collection_path))
                
                indexer.index(
                    name=self.index_name,
                    collection=collection,
                    overwrite=True,
                )
            
            logger.info("ColBERT index built successfully")
        
        # Load searcher
        logger.info("Loading ColBERT searcher...")
        
        with Run().context(RunConfig(experiment=str(index_dir.parent.name))):
            self.searcher = Searcher(
                index=self.index_name,
                collection=str(self.collection_path),
            )
        
        self._initialized = True
        logger.info(f"ColBERT retriever loaded with {len(self.texts)} passages")
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using ColBERT.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of RetrievalResult objects
        """
        self._ensure_initialized()
        
        if k is None:
            k = self.k_default
        
        try:
            # Search
            pids, ranks, scores = self.searcher.search(query, k=k)
            
            # Convert to results
            results = []
            for rank, (pid, score) in enumerate(zip(pids, scores), 1):
                # Get text from collection
                text = self.searcher.collection[pid]
                
                # Get chunk metadata
                chunk = self.chunks[pid] if self.chunks and pid < len(self.chunks) else {}
                doc_id = chunk.get("doc_id")
                chunk_id = chunk.get("chunk_id")
                metadata = {
                    "retriever": "colbert",
                    "pid": int(pid),
                    **chunk.get("metadata", {}),
                }
                
                results.append(RetrievalResult(
                    text=text,
                    score=float(score),
                    rank=rank,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    metadata=metadata,
                ))
            
            logger.debug(f"ColBERT retrieval: {len(results)} results for query '{query[:50]}'")
            return results
            
        except Exception as e:
            logger.error(f"ColBERT retrieval failed: {e}")
            raise

