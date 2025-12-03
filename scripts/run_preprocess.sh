#!/bin/bash
set -e

echo "Running RALFS Preprocessing Pipeline"

poetry run python -m src.cli \
    task=preprocess \
    data.max_samples=15 \    # 1000
    data.chunk_size=512 \
    data.overlap=128 \
    data.dataset=govreport \
    --config-path ../configs/train \
    --config-name fid_base
echo "Preprocessing Complete! Chunks saved to data/processed/govreport_chunks.json"
