# src/ralfs/evaluation/metrics.py
from rouge_score import rouge_scorer
from evaluate import load
from bert_score import score as bert_score
from ralfs.evaluation.faithfulness import compute_egf
from ralfs.core.logging import get_logger

logger = get_logger(__name__)

rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bert = load("bertscore")

def evaluate_predictions(preds, refs):
    ) -> dict:
    results = []
    for p, r in zip(preds, refs):
        rouge_scores = rouge.score(r["summary"], p["summary"])
        P, R, F = bert_score([p["summary"]], [r["summary"]], lang="en", verbose=False)
        egf = compute_egf(r["summary"], p["summary"])
        results.append({
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rouge2": rouge_scores["rouge2"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure,
            "bertscore": F.mean().item(),
            "egf": egf,
        })
    avg = {k: sum(d[k] for d in results) / len(results) for k in results[0]}
    logger.info("=== FINAL RESULTS ===")
    for k, v in avg.items():
        logger.info(f"{k:12}: {v:.4f}")
    return avg
