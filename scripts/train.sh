#!/bin/bash
set -e

echo "Starting RALFS Training"

poetry run ralfs task=train 

echo "Training complete! Checkpoints saved to checkpoints/arxiv_fid"