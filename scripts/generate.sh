#!/bin/bash
set -e

echo "Starting RALFS Generation Pipeline"

poetry run ralfs task=generate \
    +query="What are the main contributions of the Attention Is All You Need paper?" \

echo "Generation complete!"