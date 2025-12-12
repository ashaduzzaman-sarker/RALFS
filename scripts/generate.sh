#!/bin/bash
set -e
echo "Running RALFS Generation"
poetry run ralfs task=generate +query="$1"
