#!/bin/bash
# ============================================================================
# Evaluation script for RALFS
# ============================================================================

set -e

echo "📊 RALFS Evaluation Script"
echo "=========================="

# Configuration
PREDICTIONS=${1:-"results/predictions.json"}
REFERENCES=${2:-"data/test/references.json"}
OUTPUT_DIR=${3:-"results/evaluation_$(date +%Y%m%d_%H%M%S)"}
METRICS=${4:-"rouge,bertscore,egf"}

echo "Predictions: $PREDICTIONS"
echo "References: $REFERENCES"
echo "Output: $OUTPUT_DIR"
echo "Metrics: $METRICS"
echo ""

# Check files exist
if [ ! -f "$PREDICTIONS" ]; then
    echo "❌ Predictions file not found: $PREDICTIONS"
    exit 1
fi

if [ ! -f "$REFERENCES" ]; then
    echo "❌ References file not found: $REFERENCES"
    exit 1
fi

# Run evaluation
echo "Running evaluation..."
poetry run ralfs evaluate \
    $PREDICTIONS \
    $REFERENCES \
    --output-dir $OUTPUT_DIR \
    --metrics $METRICS

echo ""
echo "✅ Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"

# Display results
echo ""
echo "Summary:"
cat "${OUTPUT_DIR}/eval_aggregated.json"