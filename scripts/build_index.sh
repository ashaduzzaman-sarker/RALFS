#!/bin/bash
set -e
echo "Building FAISS + ColBERT index"
poetry run ralfs task=build_index
echo "Index ready → data/index/faiss.index + ColBERT cache"
