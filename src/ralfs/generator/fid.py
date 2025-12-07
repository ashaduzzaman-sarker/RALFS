# src/ralfs/generator/fid.py
from __future__ import annotations
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig
from ralfs.generator.base import BaseGenerator
from ralfs.core.logging import get_logger
from ralfs.core.config import RALFSConfig

logger = get_logger(__name__)

class FiDGenerator(BaseGenerator):
    def __init__(self, cfg: RALFSConfig):
        self.cfg = cfg.generator
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model.name)

        # Apply LoRA (parameter-efficient — works on T4/A100)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q", "v"],
            bias="none"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.eval()
        logger.info(f"FiD Generator loaded with LoRA r=16: {self.cfg.model.name}")

    def generate(self, query: str, passages: List[Dict]) -> Tuple[str, Dict]:
        # Adaptive k
        scores = [p["score"] for p in passages]
        k = self._adaptive_k(scores, min_k=self.cfg.adaptive.min_k, max_k=self.cfg.adaptive.max_k)

        selected = passages[:k]
        inputs = [f"question: {query} context: {p['text']}" for p in selected]
        
        encoded = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.model.max_input_length
        )
        
        with torch.no_grad():
            output = self.model.generate(
                encoded.input_ids,
                max_length=self.cfg.model.max_output_length,
                num_beams=self.cfg.model.num_beams,
                early_stopping=True
            )
        summary = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        stats = {"k_used": k, "num_passages": len(passages)}
        return summary, stats

    def _adaptive_k(self, scores: List[float], min_k: int, max_k: int) -> int:
        if len(scores) < min_k:
            return len(scores)
        drop_off = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
        max_drop = max(range(min_k, max_k), key=lambda i: drop_off[i-1] if i > 0 else 0)
        return max_drop + 1