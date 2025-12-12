# src/ralfs/evaluation/main.py
from ralfs.evaluation.metrics import evaluate_predictions
from ralfs.evaluation.human import create_human_eval_template
from ralfs.utils.io import load_json

def evaluate(cfg):
    preds = load_json("results/predictions.jsonl", as_jsonl=True)
    refs = load_json("data/govreport/test.jsonl", as_jsonl=True)  # adjust path
    evaluate_predictions(preds, refs)
    create_human_eval_template("results/predictions.jsonl")
