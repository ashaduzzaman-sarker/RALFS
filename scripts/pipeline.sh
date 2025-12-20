#!/bin/bash
# ============================================================================
# RALFS Complete Pipeline Script
# Run full pipeline: preprocess → index → train → evaluate
# ============================================================================
set -e

DATASET=${1:-"arxiv"}
MAX_SAMPLES=${2:-10}
CONFIG=${3:-"configs/train/default.yaml"}

echo "🔄 RALFS Complete Pipeline"
echo "=========================="
echo "Dataset: $DATASET"
echo "Max samples: $MAX_SAMPLES"
echo "Config: $CONFIG"
echo ""

# Run complete pipeline using CLI
poetry run ralfs pipeline \
    --dataset $DATASET \
    --max-samples $MAX_SAMPLES \
    --config $CONFIG

echo ""
echo "✅ Pipeline complete!"
