#!/bin/bash
set -e

echo "🚀 Running RALFS Preprocessing Pipeline"
poetry run python -m src.cli preprocess \
    --config configs/train/fid_base.yaml \
    data.max_samples=1000 \
    data.chunk_size=256

echo "✅ Preprocessing Complete!"