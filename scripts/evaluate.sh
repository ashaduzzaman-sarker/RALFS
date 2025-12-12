#!/bin/bash
set -e
echo "Running Evaluation"
poetry run ralfs task=evaluate
