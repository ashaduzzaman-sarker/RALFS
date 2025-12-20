#!/bin/bash
# ============================================================================
# RALFS Index Building Script
# ============================================================================
set -e

DATASET=${1:-"arxiv"}
FORCE=${2:-""}

echo "🔍 RALFS Index Building"
echo "Dataset: $DATASET"

CMD="ralfs build-index --dataset $DATASET"
[ "$FORCE" = "--force" ] && CMD="$CMD --force"

$CMD

echo "✅ Index building complete!"
