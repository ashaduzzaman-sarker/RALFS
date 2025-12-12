#!/bin/bash
set -e

echo "Starting RALFS Training"
echo "Dataset: ${1:-arxiv}"
echo "Using accelerate (fp16 on T4/A100)"

accelerate launch \
    --mixed_precision=fp16 \
    src/ralfs/training/trainer.py \
    +data.dataset=${1:-arxiv} \
    +model.batch_size=1 \
    +model.grad_accum=16 \
    +model.epochs=3

echo "Training complete! Checkpoints in checkpoints/${1:-arxiv}_fid"
