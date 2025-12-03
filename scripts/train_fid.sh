#!/bin/bash
set -e

echo "Starting FiD Training on arXiv-summarization (T4 SAFE MODE)"

accelerate launch \
    --mixed_precision=fp16 \
    --num_processes=1 \
    src/training/train_fid.py \
    model.batch_size=1 \
    model.grad_accum=16 \
    model.lr=5e-5 \
    model.epochs=1 \
    model.name="google/flan-t5-large" \
    output_dir="checkpoints/arxiv_fid_final"
