#!/bin/bash
set -e

echo "Building FAISS Index for RALFS"

poetry run python -m src.cli \
    task=build_index \
    data.dataset=govreport \
    --config-path configs/train \
    --config-name fid_base

echo "Index ready at data/index/faiss.index"