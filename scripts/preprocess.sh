#!/bin/bash
set -e
echo "Starting RALFS Preprocessing"
poetry run ralfs task=preprocess \
    data.dataset=arxiv \
    data.max_samples=1000
echo "Preprocessing complete → data/processed/arxiv_chunks.jsonl"
