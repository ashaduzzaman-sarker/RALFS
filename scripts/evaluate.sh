#!/bin/bash
poetry run python -m src.evaluation.evaluate \
    --pred results/predictions.jsonl \
    --gold data/govreport/test.jsonl \
    --output results/report.tex