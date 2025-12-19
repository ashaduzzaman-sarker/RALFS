# ============================================================================
# File: ralfs/retriever/hybrid.py
# ============================================================================
"""Hybrid retrieval combining dense, sparse, and ColBERT."""

from __future__ import annotations
from typing import List, Optional
import numpy as np

from ralfs.retriever.base import BaseRetriever, RetrievalResult
from ralfs.retriever.dense import DenseRetriever
from ralfs.retriever.sparse import SparseRetriever
from ralfs.retriever.colbert import ColBERTRetriever
from ralfs.retriever.reranker import CrossEncoderReranker
from ralfs.retriever.utils import reciprocal_rank_fusion, normalize_scores
from ralfs.core.logging import get_logger

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval combining multiple retrieval methods.
    
    Pipeline:
    1. Retrieve from Dense (FAISS), Sparse (BM25), and ColBERT
    2. Fuse results using Reciprocal Rank Fusion (RRF)
    3. Rerank top candidates with Cross-Encoder
    4. Return top-k final results
    
    Features:
    - Best-of-breed combination
    - RRF fusion (proven effective)
    - Cross-encoder reranking
    - Configurable weights and parameters
    """
    
    def __init__(self, cfg):
        """Initialize hybrid retriever."""
        super().__init__(cfg)
        
        self.cfg = cfg
        
        # Get fusion config
        fusion_config = getattr(cfg.retriever, 'fusion', None)
        if fusion_config:
            self.fusion_method = getattr(fusion_config, 'method', 'rrf')
            self.alpha = getattr(fusion_config, 'alpha', 0.4)  # Dense weight
            self.beta = getattr(fusion_config, 'beta', 0.3)   # Sparse weight
            self.gamma = getattr(fusion_config, 'gamma', 0.3)  # ColBERT weight
            self.rrf_k = getattr(fusion_config, 'rrf_k', 60)
        else:
            self.fusion_method = 'rrf'
            self.alpha = 0.4
            self.beta = 0.3
            self.gamma = 0.3
            self.rrf_k = 60
        
        # Verify weights sum to 1.0 (approximately)
        weight_sum = self.alpha + self.beta + self.gamma
        if not (0.99 <= weight_sum <= 1.01):
            logger.warning(
                f"Fusion weights don't sum to 1.0: "
                f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma} "
                f"(sum={weight_sum})"
            )
        
        # Initialize retrievers
        logger.info("Initializing hybrid retriever components...")
        
        self.dense = DenseRetriever(cfg)
        self.sparse = SparseRetriever(cfg)
        
        # ColBERT is optional (heavy dependency)
        try:
            self.colbert = ColBERTRetriever(cfg)
            self.use_colbert = True
        except ImportError:
            logger.warning(
                "ColBERT not available. Install with: "
                "pip install git+https://github.com/stanford-nlp/ColBERT.git"
            )
            self.colbert = None
            self.use_colbert = False
        
        # Reranker
        self.reranker = CrossEncoderReranker(cfg)
        
        logger.info("Hybrid retriever initialized")
    
    def load_index(self, index_path: Optional[str] = None) -> None:
        """
        Load all retriever indexes.
        
        Args:
            index_path: Not used (each retriever uses its own path)
        """
        logger.info("Loading indexes for hybrid retriever...")
        
        # Load dense index
        try:
            self.dense.load_index()
            logger.info("✓ Dense index loaded")
        except Exception as e:
            logger.error(f"Failed to load dense index: {e}")
            raise
        
        # Load sparse index (builds BM25 from chunks)
        try:
            self.sparse.load_index()
            logger.info("✓ Sparse index loaded")
        except Exception as e:
            logger.error(f"Failed to load sparse index: {e}")
            raise
        
        # Load ColBERT index (optional)
        if self.use_colbert and self.colbert:
            try:
                self.colbert.load_index()
                logger.info("✓ ColBERT index loaded")
            except Exception as e:
                logger.warning(f"ColBERT index loading failed: {e}")
                logger.warning("Continuing without ColBERT")
                self.use_colbert = False
        
        self._initialized = True
        logger.info("All indexes loaded successfully")
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve using hybrid approach: Dense + Sparse + ColBERT + Rerank.
        
        Args:
            query: Search query
            k: Final number of results to return
        
        Returns:
            List of reranked RetrievalResult objects
        """
        self._ensure_initialized()
        
        # Get k values from config
        reranker_config = getattr(self.cfg.retriever, 'reranker', None)
        if k is None:
            if reranker_config:
                k = getattr(reranker_config, 'top_k', 20)
            else:
                k = getattr(self.cfg.retriever, 'k_final', 20)
        
        # Get intermediate k for reranking
        if reranker_config:
            k_rerank = getattr(reranker_config, 'k_rerank', 50)
        else:
            k_rerank = getattr(self.cfg.retriever, 'k_rerank', 50)
        
        logger.info(f"Hybrid retrieval for query: '{query[:50]}...'")
        
        # Step 1: Retrieve from each method
        results_list = []
        
        # Dense retrieval
        try:
            dense_results = self.dense.retrieve(query)
            results_list.append(dense_results)
            logger.debug(f"Dense: {len(dense_results)} results")
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            raise
        
        # Sparse retrieval
        try:
            sparse_results = self.sparse.retrieve(query)
            results_list.append(sparse_results)
            logger.debug(f"Sparse: {len(sparse_results)} results")
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            raise
        
        # ColBERT retrieval (optional)
        if self.use_colbert and self.colbert:
            try:
                colbert_results = self.colbert.retrieve(query)
                results_list.append(colbert_results)
                logger.debug(f"ColBERT: {len(colbert_results)} results")
            except Exception as e:
                logger.warning(f"ColBERT retrieval failed: {e}")
        
        # Step 2: Fuse results
        if self.fusion_method == 'rrf':
            # Reciprocal Rank Fusion (recommended)
            fused_results = reciprocal_rank_fusion(results_list, k=self.rrf_k)
        elif self.fusion_method == 'linear':
            # Linear combination of normalized scores
            fused_results = self._linear_fusion(
                dense_results,
                sparse_results,
                colbert_results if self.use_colbert else None,
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        logger.debug(f"Fused to {len(fused_results)} results")
        
        # Step 3: Take top candidates for reranking
        candidates = fused_results[:k_rerank]
        
        # # Step 4: Rerank with cross-encoder
        # reranked_results = self.reranker.rerank(query, candidates, top_k=k)

        # # Ensure metadata is set
        # for result in reranked_results:
        #     if result.metadata is None:
        #         result.metadata = {}
        #     result.metadata["reranked"] = True

        # logger.info(
        #     f"Hybrid retrieval complete: {len(reranked_results)} final results "
        #     f"(from {len(fused_results)} fused candidates)"
        # )
        # =====
        # Step 4: Rerank with cross-encoder
        reranked_results = self.reranker.rerank(query, candidates, top_k=k)

        # Normalize final scores to [0, 1] range
        # from ralfs.retriever.utils import normalize_scores
        if reranked_results:
            scores = [r.score for r in reranked_results]
            normalized = normalize_scores(scores)
            for result, norm_score in zip(reranked_results, normalized):
                result.score = norm_score

        return reranked_results
    
    def _linear_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        colbert_results: Optional[List[RetrievalResult]] = None,
    ) -> List[RetrievalResult]:
        """
        Fuse results using weighted linear combination.
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            colbert_results: Results from ColBERT (optional)
        
        Returns:
            Fused results
        """
        # Normalize scores for each retriever
        dense_scores = normalize_scores([r.score for r in dense_results])
        sparse_scores = normalize_scores([r.score for r in sparse_results])
        
        # Build text -> score mapping
        fused_scores: Dict[str, float] = {}
        text_to_result: Dict[str, RetrievalResult] = {}
        
        # Add dense scores
        for result, norm_score in zip(dense_results, dense_scores):
            text = result.text
            fused_scores[text] = self.alpha * norm_score
            text_to_result[text] = result
        
        # Add sparse scores
        for result, norm_score in zip(sparse_results, sparse_scores):
            text = result.text
            if text in fused_scores:
                fused_scores[text] += self.beta * norm_score
            else:
                fused_scores[text] = self.beta * norm_score
                text_to_result[text] = result
        
        # Add ColBERT scores (if available)
        if colbert_results:
            colbert_scores = normalize_scores([r.score for r in colbert_results])
            for result, norm_score in zip(colbert_results, colbert_scores):
                text = result.text
                if text in fused_scores:
                    fused_scores[text] += self.gamma * norm_score
                else:
                    fused_scores[text] = self.gamma * norm_score
                    text_to_result[text] = result
        
        # Sort by fused score
        sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Convert to results
        fused_results = []
        for rank, (text, score) in enumerate(sorted_items, 1):
            result = text_to_result[text]
            result.score = score
            result.rank = rank
            if result.metadata:
                result.metadata["fusion"] = "linear"
            fused_results.append(result)
        
        return fused_results