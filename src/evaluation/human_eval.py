# src/evaluation/human_eval.py
import json
import csv
from pathlib import Path

def generate_annotation_template(predictions_path: str, output_csv: str = "human_eval.csv", n_samples: int = 50):
    Path(output_csv).parent.mkdir(exist_ok=True)
    samples = []
    with open(predictions_path) as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            item = json.loads(line.strip())
            samples.append(item)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Title", "Reference Summary", "Model Output", "Faithfulness", "Coherence", "Comments"])
        for i, item in enumerate(samples, 1):
            writer.writerow([
                i,
                item.get("title", "")[:100],
                item["gold"][:1500].replace("\n", " "),
                item["pred"][:1500].replace("\n", " "),
                "", "", ""
            ])
    print(f"Human evaluation template ready: {output_csv}")