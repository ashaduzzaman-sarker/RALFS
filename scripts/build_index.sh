#!/bin/bash
set -e

echo "RALFS v1.0 — Building Dense FAISS Index"

poetry run ralfs task=build_index \

echo "FAISS index built successfully! Index saved: data/index/faiss.index"
echo "Metadata saved: data/index/faiss.metadata.json"