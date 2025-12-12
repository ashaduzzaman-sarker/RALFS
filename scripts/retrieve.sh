#!/bin/bash
set -e

echo "Starting RALFS Retriever Pipeline"

poetry run ralfs task=retrieve \
    query="attention is all you need" \

echo "Retriever complete!"
