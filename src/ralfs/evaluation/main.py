# src/ralfs/evaluation/main.py
from ralfs.evaluation.metrics import evaluate_predictions
from ralfs.evaluation.human import create_human_eval_template
from ralfs.utils.io import load_json
from ralfs.core.logging import get_logger

logger = get_logger(__name__)

def evaluate(cfg):
    # Dummy data for now — replace with real paths
    preds = load_json("results/predictions.jsonl", as_jsonl=True)
    refs = load_json("data/govreport/test.jsonl", as_jsonl=True)[:len(preds)]
    evaluate_predictions(preds, refs)
    create_human_eval_template("results/predictions.jsonl")
