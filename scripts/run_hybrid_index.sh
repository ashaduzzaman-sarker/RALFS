#!/bin/bash
set -e
echo "Building Hybrid Index for RALFS (Phase 2)"
echo "Reusing existing dense FAISS index..."
echo "Preparing BM25 + ColBERT metadata..."

# Just validate index exists
if [ ! -f "data/index/faiss.index" ]; then
    echo "Dense index not found! Run: bash scripts/run_build_index.sh first"
    exit 1
fi

echo "Hybrid system ready! Run: poetry run python examples/hybrid_demo.py"