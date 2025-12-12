# src/ralfs/evaluation/metrics.py
from rouge_score import rouge_scorer
from evaluate import load
from bert_score import score as bert_score
from ralfs.evaluation.faithfulness import compute_egf
from ralfs.core.logging import get_logger

logger = get_logger(__name__)

rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bert_scorer = load("bertscore")

def evaluate_predictions(predictions: List[Dict], references: List[Dict]) -> Dict:
    results = []
    for pred, ref in zip(predictions, references):
        r = rouge_scorer.score(ref["summary"], pred["summary"])
        P, R, F1 = bert_score([pred["summary"]], [ref["summary"]], lang="en", verbose=False)
        egf = compute_egf(ref["summary"], pred["summary"])

        results.append({
            "rouge1": r["rouge1"].fmeasure,
            "rouge2": r["rouge2"].fmeasure,
            "rougeL": r["rougeL"].fmeasure,
            "bertscore_f1": F1.mean().item(),
            "egf": egf
        })

    avg = {k: sum(r[k] for r in results) / len(results) for k in results[0]}
    logger.info("Evaluation Results:")
    for k, v in avg.items():
        logger.info(f"  {k.upper():12} {v:.4f}")
    return avg
