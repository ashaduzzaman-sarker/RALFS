# src/ralfs/evaluation/faithfulness.py
import spacy
from typing import List, Dict
from collections import Counter
from ralfs.core.logging import get_logger

logger = get_logger(__name__)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def compute_egf(gold: str, pred: str) -> float:
    """Entity Grid Faithfulness (Barzilay & Lapata, 2008)"""
    def get_entities(text):
        doc = nlp(text)
        entities = []
        for sent in doc.sents:
            sent_ents = {ent.text.lower() for ent in sent.ents}
            entities.append(sent_ents)
        return entities

    gold_grid = get_entities(gold)
    pred_grid = get_entities(pred)

    total = 0
    matched = 0
    for gold_sent in gold_grid:
        total += len(gold_sent)
        for pred_sent in pred_grid:
            matched += len(gold_sent & pred_sent)

    return matched / total if total > 0 else 0.0
