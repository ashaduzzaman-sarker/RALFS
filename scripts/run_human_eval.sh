#!/bin/bash
# ============================================================================
# RALFS Human Evaluation Template Creation Script
# ============================================================================
set -e

PREDICTIONS=${1:-"results/predictions.jsonl"}
REFERENCES=${2:-"data/test/references.jsonl"}
OUTPUT=${3:-"results/human_eval.csv"}
NUM_SAMPLES=${4:-50}

echo "👥 RALFS Human Evaluation Template"
echo "Predictions: $PREDICTIONS"
echo "References: $REFERENCES"
echo "Output: $OUTPUT"
echo "Samples: $NUM_SAMPLES"

poetry run ralfs human-eval $PREDICTIONS $REFERENCES \
    --output-file $OUTPUT \
    --num-samples $NUM_SAMPLES

echo "✅ Template created: $OUTPUT"
echo "Send this file to human annotators"