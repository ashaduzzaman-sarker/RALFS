#!/bin/bash
set -e

echo "Starting RALFS Preprocessing Pipeline"

poetry run ralfs task=preprocess \
    +data.max_samples=100

echo "Preprocessing complete!"