#!/bin/bash
echo "Preparing human evaluation files..."
poetry run python -m src.evaluation.human_eval \
    --predictions results/predictions.jsonl \
    --output human_eval_batch1.csv

echo "Done! Send human_eval_batch1.csv to annotators"