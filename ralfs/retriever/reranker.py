# ============================================================================
# File: ralfs/retriever/reranker.py
# ============================================================================
"""Cross-encoder reranking."""

from __future__ import annotations
from typing import List, Optional, Union
from sentence_transformers import CrossEncoder

from ralfs.retriever.base import RetrievalResult
from ralfs.core.logging import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker for fine-grained relevance scoring.
    
    Features:
    - Cross-encoder architecture (query-document attention)
    - More accurate than bi-encoder retrieval
    - Used as second-stage reranker
    """
    
    def __init__(self, cfg):
        """Initialize reranker."""
        self.cfg = cfg
        
        # Get config
        reranker_config = getattr(cfg.retriever, 'reranker', None)
        if reranker_config:
            model_name = getattr(reranker_config, 'model', 'cross-encoder/ms-marco-MiniLM-L-12-v2')
            self.top_k = getattr(reranker_config, 'top_k', 20)
            self.batch_size = getattr(reranker_config, 'batch_size', 16)
            # self.enabled = getattr(reranker_config, 'enabled', True)
        
            if isinstance(reranker_config, dict):
                self.enabled = reranker_config.get('enabled', True)
            else:
                self.enabled = getattr(reranker_config, 'enabled', True)
        else:
            model_name = getattr(cfg.retriever, 'reranker_model', 'cross-encoder/ms-marco-MiniLM-L-12-v2')
            self.top_k = getattr(cfg.retriever, 'k_final', 20)
            self.batch_size = 16
            self.enabled = True
        
        if self.enabled:
            logger.info(f"Loading reranker model: {model_name}")
            self.model = CrossEncoder(model_name, max_length=512)
            logger.info("Reranker loaded successfully")
        else:
            logger.info("Reranker disabled")
            self.model = None
    
    def rerank(
        self,
        query: str,
        candidates: List[Union[RetrievalResult, dict]],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: Search query
            candidates: List of candidates (RetrievalResult or dict)
            top_k: Number of results to return
        
        Returns:
            Reranked list of RetrievalResult objects
        """
        if not self.enabled or self.model is None:
            # Return candidates as-is
            logger.debug("Reranker disabled, returning candidates unchanged")
            if isinstance(candidates[0], dict):
                return [RetrievalResult(**c) for c in candidates]
            return candidates
        
        if top_k is None:
            top_k = self.top_k
        
        if not candidates:
            return []
        
        try:
            # Prepare pairs
            pairs = []
            for candidate in candidates:
                if isinstance(candidate, RetrievalResult):
                    text = candidate.text
                elif isinstance(candidate, dict):
                    text = candidate["text"]
                else:
                    text = str(candidate)
                pairs.append([query, text])
            
            # Score with cross-encoder
            logger.debug(f"Reranking {len(candidates)} candidates...")
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            
            # Combine and sort
            scored_candidates = list(zip(candidates, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Take top-k and convert to RetrievalResult
            results = []
            for rank, (candidate, score) in enumerate(scored_candidates[:top_k], 1):
                if isinstance(candidate, RetrievalResult):
                    result = candidate
                    result.score = float(score)
                    result.rank = rank
                    if result.metadata is None: 
                        result.metadata = {}
                    result.metadata["reranked"] = True
                    result.metadata["original_score"] = result.score
                                  
                elif isinstance(candidate, dict):
                    result = RetrievalResult(
                        text=candidate["text"],
                        score=float(score),
                        rank=rank,
                        doc_id=candidate.get("doc_id"),
                        chunk_id=candidate.get("chunk_id"),
                        metadata={
                            **candidate.get("metadata", {}),
                            "reranked": True,
                        },
                    )
                else:
                    continue
                
                results.append(result)
            
            logger.debug(f"Reranked to top-{len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            raise
