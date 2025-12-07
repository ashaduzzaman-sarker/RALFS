#!/bin/bash
set -e

echo "Starting RALFS Generation Pipeline"

poetry run ralfs task=generate \
    +query="WWhat are the current challenges in quantum computing research?" \

echo "Generation complete!"
