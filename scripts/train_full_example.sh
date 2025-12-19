#!/bin/bash
# ============================================================================
# Complete training example with all steps
# ============================================================================

set -e

echo "🚀 RALFS Complete Training Pipeline"
echo "===================================="

# Configuration
DATASET="arxiv"
MAX_SAMPLES=100  # Use more for real training
CONFIG="configs/train/default.yaml"
OUTPUT_DIR="checkpoints/example_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Max samples: $MAX_SAMPLES"
echo "  Config: $CONFIG"
echo "  Output: $OUTPUT_DIR"
echo ""

# Step 1: Preprocess data
echo "Step 1/5: Preprocessing data..."
ralfs preprocess \
    --dataset $DATASET \
    --max-samples $MAX_SAMPLES \
    --force-download

# Step 2: Build indexes
echo ""
echo "Step 2/5: Building indexes..."
ralfs build-index \
    --dataset $DATASET \
    --force

# Step 3: Test retrieval
echo ""
echo "Step 3/5: Testing retrieval..."
ralfs search "deep learning neural networks" \
    --dataset $DATASET \
    --k 5

# Step 4: Train model
echo ""
echo "Step 4/5: Training model..."
ralfs train \
    --config $CONFIG \
    --dataset $DATASET \
    --output-dir $OUTPUT_DIR

# Step 5: Evaluate
echo ""
echo "Step 5/5: Evaluating..."
# Note: This assumes you have test data prepared
if [ -f "data/test/${DATASET}_predictions.json" ]; then
    ralfs evaluate \
        "data/test/${DATASET}_predictions.json" \
        "data/test/${DATASET}_references.json" \
        --output-dir "${OUTPUT_DIR}/evaluation"
else
    echo "⚠️  Test data not found, skipping evaluation"
fi

echo ""
echo "✅ Complete pipeline finished!"
echo "Model saved to: $OUTPUT_DIR"