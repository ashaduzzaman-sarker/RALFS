"""Adaptive Fusion-in-Decoder: dynamically selects top-k per query."""
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class AdaptiveFiD:
    def __init__(self, model_name: str = "google/flan-t5-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def generate(self, query: str, passages: List[Dict], min_k: int = 5, max_k: int = 40) -> str:
        texts = [p["text"] for p in passages]
        scores = [p["score"] for p in passages]

        # Adaptive k: start low, grow until score drop-off
        best_k = min_k
        best_score_drop = float('inf')
        for k in range(min_k, min(max_k, len(texts)) + 1):
            current_drop = scores[k-1] - scores[min(k, len(scores)-1)]
            if current_drop < best_score_drop:
                best_k = k
                best_score_drop = current_drop
            else:
                break  # score plateau → stop

        selected = texts[:best_k]
        inputs = [f"Question: {query} Context: {ctx}" for ctx in selected]
        encoded = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            output = self.model.generate(
                encoded.input_ids,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
        summary = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return summary, best_k