# ============================================================================
# File: ralfs/generator/fid.py
# ============================================================================
"""Fusion-in-Decoder generator with adaptive k selection."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, PeftModel

from ralfs.generator.base import BaseGenerator, GenerationResult
from ralfs.generator.adaptive_k import AdaptiveKSelector
from ralfs.core.logging import get_logger

logger = get_logger(__name__)


class FiDGenerator(BaseGenerator):
    """
    Fusion-in-Decoder generator with adaptive k selection.
    
    Features:
    - Adaptive k selection (novel contribution)
    - LoRA for efficient fine-tuning
    - Proper FiD input formatting
    - Batch processing support
    - Configurable generation parameters
    """
    
    def __init__(self, cfg):
        """Initialize FiD generator."""
        super().__init__(cfg)
        
        # Get generator config
        gen_config = cfg.generator
        self.model_config = getattr(gen_config, 'model', gen_config)
        self.adaptive_config = getattr(gen_config, 'adaptive', None)
        self.lora_config = getattr(gen_config, 'lora', None)
        
        # Model name
        self.model_name = getattr(self.model_config, 'name', 'google/flan-t5-large')
        
        # Generation parameters
        self.max_input_length = getattr(self.model_config, 'max_input_length', 512)
        self.max_output_length = getattr(self.model_config, 'max_output_length', 200)
        self.num_beams = getattr(self.model_config, 'num_beams', 4)
        self.length_penalty = getattr(self.model_config, 'length_penalty', 1.0)
        self.no_repeat_ngram_size = getattr(self.model_config, 'no_repeat_ngram_size', 3)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self._load_model()
        
        # Initialize adaptive k selector
        self._init_adaptive_k()
        
        self._initialized = True
        logger.info(f"FiD generator initialized on {self.device}")
    
    def _load_model(self):
        """Load model and tokenizer with optional LoRA."""
        logger.info(f"Loading model: {self.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            )
            
            # Apply LoRA if configured
            if self.lora_config and getattr(self.lora_config, 'enabled', True):
                logger.info("Applying LoRA configuration...")
                
                lora_r = getattr(self.lora_config, 'r', 16)
                lora_alpha = getattr(self.lora_config, 'alpha', 32)
                lora_dropout = getattr(self.lora_config, 'dropout', 0.1)
                target_modules = getattr(self.lora_config, 'target_modules', ['q', 'v'])
                
                peft_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                    bias="none",
                    task_type="SEQ_2_SEQ_LM",
                )
                
                self.model = get_peft_model(self.model, peft_config)
                logger.info(f"LoRA applied: r={lora_r}, alpha={lora_alpha}")
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _init_adaptive_k(self):
        """Initialize adaptive k selector."""
        gen_config = self.cfg.generator
        
        # Check for max_k at generator level (for tests)
        max_k_override = getattr(gen_config, 'max_k', None)
        
        if self.adaptive_config:
            enabled = getattr(self.adaptive_config, 'enabled', True)
            
            if enabled:
                min_k = getattr(self.adaptive_config, 'min_k', 5)
                max_k = max_k_override or getattr(self.adaptive_config, 'max_k', 30)
                default_k = getattr(self.adaptive_config, 'default_k', 20)
                strategy = getattr(self.adaptive_config, 'strategy', 'score_dropoff')
                
                self.adaptive_k_selector = AdaptiveKSelector(
                    min_k=min_k,
                    max_k=max_k,
                    default_k=default_k,
                    strategy=strategy,
                )
                self.use_adaptive_k = True
                logger.info(f"Adaptive k enabled: strategy={strategy}")
            else:
                self.use_adaptive_k = False
                self.fixed_k = getattr(self.adaptive_config, 'default_k', 20)
                logger.info(f"Adaptive k disabled, using fixed k={self.fixed_k}")
        else:
            # Check if max_k is set at generator config level
            min_k = getattr(gen_config, 'adaptive_k_min', 5)
            max_k = max_k_override or getattr(gen_config, 'adaptive_k_max', 30)
            default_k = min(20, max_k)  # Ensure default_k <= max_k
            
            # Default: use adaptive k
            self.adaptive_k_selector = AdaptiveKSelector(
                min_k=min_k,
                max_k=max_k,
                default_k=default_k,
            )
            self.use_adaptive_k = True
    
    def generate(
        self,
        query: str,
        passages: List[Dict[str, Any]],
    ) -> GenerationResult:
        """
        Generate summary using FiD with adaptive k.
        
        Args:
            query: Input query/question
            passages: Retrieved passages with 'text' and 'score' fields
        
        Returns:
            GenerationResult object
        """
        self._ensure_initialized()
        
        if not passages:
            return GenerationResult(
                summary="No relevant passages found.",
                query=query,
                k_used=0,
                num_passages=0,
            )
        
        start_time = time.time()
        
        try:
            # Select k adaptively
            scores = [p.get("score", 0.0) for p in passages]
            
            if self.use_adaptive_k:
                k = self.adaptive_k_selector.select_k(scores)
                strategy = self.adaptive_k_selector.strategy
            else:
                k = min(self.fixed_k, len(passages))
                strategy = "fixed"
            
            # Select top-k passages
            selected_passages = passages[:k]
            
            # Format inputs for FiD
            inputs = self._format_fid_inputs(query, selected_passages)
            
            # Tokenize
            encoded = self.tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_length,
            )
            
            # Move to device
            encoded = {key: val.to(self.device) for key, val in encoded.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **encoded,
                    max_length=self.max_output_length,
                    num_beams=self.num_beams,
                    length_penalty=self.length_penalty,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                    early_stopping=True,
                )
            
            # Decode
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            
            # Create result
            result = GenerationResult(
                summary=summary,
                query=query,
                k_used=k,
                num_passages=len(passages),
                adaptive_strategy=strategy,
                generation_time=generation_time,
                metadata={
                    "model": self.model_name,
                    "scores": scores[:k],
                    "avg_score": sum(scores[:k]) / k if k > 0 else 0.0,
                },
            )
            
            logger.info(
                f"Generated summary: k={k}/{len(passages)}, "
                f"strategy={strategy}, time={generation_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _format_fid_inputs(
        self,
        query: str,
        passages: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Format inputs for FiD.
        
        FiD format: "question: {query} context: {passage_text}"
        
        Args:
            query: Input query
            passages: List of passages
        
        Returns:
            List of formatted input strings
        """
        inputs = []
        for passage in passages:
            text = passage.get("text", "")
            # FiD concatenates query with each passage
            formatted = f"question: {query} context: {text}"
            inputs.append(formatted)
        
        return inputs
    
    def generate_batch(
        self,
        queries: List[str],
        passages_list: List[List[Dict[str, Any]]],
    ) -> List[GenerationResult]:
        """
        Generate summaries for multiple queries (batch processing).
        
        Args:
            queries: List of queries
            passages_list: List of passage lists (one per query)
        
        Returns:
            List of GenerationResult objects
        """
        if len(queries) != len(passages_list):
            raise ValueError(
                f"Length mismatch: {len(queries)} queries vs "
                f"{len(passages_list)} passage lists"
            )
        
        results = []
        for query, passages in zip(queries, passages_list):
            result = self.generate(query, passages)
            results.append(result)
        
        return results

