# src/ralfs/evaluation/human.py
import csv
from typing import List, Dict
from pathlib import Path
from ralfs.utils.io import load_json
from ralfs.core.logging import get_logger

logger = get_logger(__name__)

def create_human_eval_template(pred_path: str, n_samples: int = 50):
    preds = load_json(pred_path, as_jsonl=True)[:n_samples]
    out_path = Path("human_eval.csv")
    
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Query", "Reference", "Generated", "Faithfulness (1-5)", "Fluency (1-5)", "Coverage (1-5)", "Comments"])
        for i, item in enumerate(preds, 1):
            writer.writerow([
                i,
                item.get("query", "")[:200],
                item.get("gold", "")[:500],
                item["summary"][:500],
                "", "", "", ""
            ])
    logger.info(f"Human eval template created: {out_path}")
