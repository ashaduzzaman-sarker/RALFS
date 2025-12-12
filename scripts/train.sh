#!/bin/bash
set -e
echo "Starting RALFS Training"
accelerate launch --mixed_precision=fp16 src/ralfs/training/trainer.py
