#!/bin/bash
set -e

echo "Starting RALFS Preprocessing Pipeline"

poetry run ralfs task=preprocess \
    +data.dataset=arxiv \
    +data.max_samples=10 \
    +data.chunk_size=512 \
    +data.overlap=128

echo "Preprocessing complete! Chunks saved to data/processed/arxiv_chunks.jsonl"