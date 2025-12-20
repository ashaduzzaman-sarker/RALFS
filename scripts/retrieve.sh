#!/bin/bash
# ============================================================================
# RALFS Retrieval Script
# ============================================================================
set -e

QUERY=${1:-"attention is all you need"}
DATASET=${2:-"arxiv"}
K=${3:-10}

echo "🔍 RALFS Retrieval"
echo "Query: $QUERY"
echo "Dataset: $DATASET"
echo "Top K: $K"

poetry run ralfs search "$QUERY" \
    --dataset $DATASET \
    --k $K

echo "✅ Retrieval complete!"
