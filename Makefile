# ============================================================================
# Makefile for RALFS
# ============================================================================

.PHONY: help install test lint format clean docs

# Default target
help:
	@echo "RALFS Makefile"
	@echo "=============="
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install dependencies"
	@echo "  test         - Run tests"
	@echo "  test-fast    - Run fast tests only"
	@echo "  test-all     - Run all tests including slow ones"
	@echo "  lint         - Run linters"
	@echo "  format       - Format code"
	@echo "  clean        - Clean generated files"
	@echo "  preprocess   - Preprocess data"
	@echo "  build-index  - Build retrieval indexes"
	@echo "  train        - Train model"
	@echo "  evaluate     - Run evaluation"

# Installation
install:
	@echo "Installing RALFS with Poetry..."
	poetry install
	poetry run pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz
	poetry run pip install git+https://github.com/stanford-futuredata/ColBERT.git
	@echo "✓ Installation complete"

install-dev:
	@echo "Installing RALFS with dev dependencies using Poetry..."
	poetry install --with dev
	poetry run pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz
	poetry run pip install git+https://github.com/stanford-futuredata/ColBERT.git
	poetry run pre-commit install
	@echo "✓ Dev installation complete"

# Testing
test:
	@echo "Running tests..."
	poetry run pytest tests/ -v -m "not slow"

test-fast:
	@echo "Running fast tests..."
	poetry run pytest tests/ -v -m "not slow and not integration"

test-all:
	@echo "Running all tests (including slow)..."
	poetry run pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	poetry run pytest tests/ -v --cov=ralfs --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

# Linting and formatting
lint:
	@echo "Running linters..."
	poetry run ruff check ralfs/ tests/
	poetry run mypy ralfs/
	@echo "✓ Linting complete"

format:
	@echo "Formatting code..."
	poetry run black ralfs/ tests/
	poetry run isort ralfs/ tests/
	@echo "✓ Formatting complete"

# Cleaning
clean:
	@echo "Cleaning generated files..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage
	@echo "✓ Cleanup complete"

# RALFS commands
preprocess:
	@echo "Preprocessing data..."
	poetry run ralfs preprocess --dataset arxiv --max-samples 10

build-index:
	@echo "Building indexes..."
	poetry run ralfs build-index --dataset arxiv

train:
	@echo "Training model..."
	poetry run ralfs train --dataset arxiv --config configs/train/debug.yaml

evaluate:
	@echo "Running evaluation..."
	poetry run ralfs evaluate results/predictions.json data/test/references.json

# Quick start
quickstart:
	@echo "Running quick start..."
	bash scripts/quickstart.sh
