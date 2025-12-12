# src/ralfs/evaluation/faithfulness.py
import spacy
from collections import Counter
from ralfs.core.logging import get_logger

logger = get_logger(__name__)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def compute_egf(gold: str, pred: str) -> float:
    """Entity Grid Faithfulness (Barzilay & Lapata 2008)"""
    def get_entities(text):
        return [[ent.text.lower() for ent in sent.ents] for sent in nlp(text).sents]

    gold_grid = get_entities(gold)
    pred_grid = get_entities(pred)

    matched = total = 0
    for g_sent in gold_grid:
        total += len(g_sent)
        for p_sent in pred_grid:
            matched += len(set(g_sent) & set(p_sent))

    return matched / total if total > 0 else 0.0
