#!/bin/bash
# ============================================================================
# Training script for RALFS
# ============================================================================

set -e  # Exit on error

echo "🚀 RALFS Training Script"
echo "========================"

# Configuration
DATASET=${1:-"arxiv"}
CONFIG=${2:-"configs/train/default.yaml"}
OUTPUT_DIR=${3:-"checkpoints/${DATASET}_$(date +%Y%m%d_%H%M%S)"}
WANDB_PROJECT=${4:-"ralfs-experiments"}

echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "W&B Project: $WANDB_PROJECT"
echo ""

# Check if data is preprocessed
if [ ! -f "data/processed/${DATASET}_train_chunks.jsonl" ]; then
    echo "❌ Preprocessed data not found. Run preprocessing first:"
    echo "   poetry run ralfs preprocess --dataset $DATASET"
    exit 1
fi

# Check if indexes are built
if [ ! -f "data/index/${DATASET}/faiss.index" ]; then
    echo "⚠️  Indexes not found. Building indexes..."
    poetry run ralfs build-index --dataset $DATASET
fi

# Start training
echo "Starting training..."
poetry run ralfs train \
    --config $CONFIG \
    --dataset $DATASET \
    --output-dir $OUTPUT_DIR \
    --wandb-project $WANDB_PROJECT

echo ""
echo "✅ Training complete!"
echo "Checkpoints saved to: $OUTPUT_DIR"