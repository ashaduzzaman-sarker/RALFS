#!/bin/bash
# ============================================================================
# RALFS Generation Script
# ============================================================================
set -e

INPUT=${1:-"data/test/documents.jsonl"}
CHECKPOINT=${2:-"checkpoints/best_model"}
OUTPUT=${3:-"results/summaries.json"}
DATASET=${4:-"arxiv"}

echo "✨ RALFS Generation"
echo "Input: $INPUT"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT"

ralfs generate $INPUT \
    --checkpoint $CHECKPOINT \
    --output-file $OUTPUT \
    --dataset $DATASET

echo "✅ Generation complete!"
