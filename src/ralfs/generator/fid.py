# src/ralfs/generator/fid.py
from __future__ import annotations
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig
from ralfs.generator.base import BaseGenerator
from ralfs.core.logging import get_logger

logger = get_logger(__name__)

class FiDGenerator(BaseGenerator):
    def __init__(self, cfg):
        self.cfg = cfg.generator
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model.name)
        self.model.to(self.device)  # ← CRITICAL: Move model to device

        # LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q", "v"],
            bias="none"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.eval()
        logger.info(f"FiD Generator loaded with LoRA r=16 on {self.device}: {self.cfg.model.name}")

    def generate(self, query: str, passages: List[Dict]) -> Tuple[str, Dict]:
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
        # ← CRITICAL: Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_length=self.cfg.model.max_output_length,
                num_beams=self.cfg.model.num_beams,
                early_stopping=True
            )
        summary = self.tokenizer.decode(output[0], skip_special_tokens=True)

        stats = {"k_used": k, "num_passages": len(passages)}
        return summary, stats

    def _adaptive_k(self, scores: List[float], min_k: int, max_k: int) -> int:
        if len(scores) <= min_k:
            return len(scores)
        drop_off = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
        best_k = min_k
        best_drop = 0
        for i in range(min_k, min(max_k, len(drop_off))):
            if drop_off[i] > best_drop:
                best_drop = drop_off[i]
                best_k = i + 1
        return best_k