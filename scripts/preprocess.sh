#!/bin/bash
set -e

echo "Starting RALFS Preprocessing Pipeline"

poetry run ralfs task=retrieve \

echo "Preprocessing complete! Chunks saved to data/processed/arxiv_chunks.jsonl"