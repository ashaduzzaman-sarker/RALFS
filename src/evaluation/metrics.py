# src/evaluation/metrics.py
from rouge_score import rouge_scorer
from evaluate import load
from src.evaluation.faithfulness import compute_egf
import bert_score

rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bert_scorer = bert_score.scorer.BertScorer(lang="en")

def compute_all_metrics(gold: str, pred: str):
    r = rouge.score(gold, pred)
    b = bert_scorer.score([pred], [gold])
    
    return {
        "rouge1": r["rouge1"].fmeasure,
        "rouge2": r["rouge2"].fmeasure,
        "rougeL": r["rougeL"].fmeasure,
        "bertscore_f1": b[2].mean().item(),
        "egf_faithfulness": compute_egf(gold, pred),
    }