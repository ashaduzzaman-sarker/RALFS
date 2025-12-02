#!/bin/bash
set -e
echo "Starting FiD Training on GovReport"

accelerate launch \
    --mixed_precision=bf16 \
    --num_processes=8 \
    --main_process_port=29501 \
    src/training/train_fid.py \
    --config configs/train/fid_train.yaml