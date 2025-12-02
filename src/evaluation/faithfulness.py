# src/evaluation/faithfulness.py
import spacy
from collections import defaultdict

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def compute_egf(gold: str, pred: str) -> float:
    def build_grid(text):
        grid = defaultdict(list)
        for sent in nlp(text).sents:
            entities = {e.text.lower() for e in sent.ents}
            for e in entities:
                grid[e].append(sent.text)
        return grid

    gold_grid = build_grid(gold)
    pred_grid = build_grid(pred)

    overlap = sum(len(set(gold_grid[e]) & set(pred_grid.get(e, []))) for e in gold_grid)
    total = sum(len(gold_grid[e]) for e in gold_grid)
    return overlap / total if total > 0 else 0.0