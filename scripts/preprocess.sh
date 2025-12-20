#!/bin/bash
# ============================================================================
# RALFS Preprocessing Script
# ============================================================================
set -e

DATASET=${1:-"arxiv"}
SPLIT=${2:-"train"}
MAX_SAMPLES=${3:-""}

echo "📦 RALFS Preprocessing"
echo "Dataset: $DATASET"
echo "Split: $SPLIT"
[ -n "$MAX_SAMPLES" ] && echo "Max samples: $MAX_SAMPLES"

CMD="ralfs preprocess --dataset $DATASET --split $SPLIT"
[ -n "$MAX_SAMPLES" ] && CMD="$CMD --max-samples $MAX_SAMPLES"

$CMD

echo "✅ Preprocessing complete!"
