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
	@echo "Installing RALFS..."
	pip install -e .
	python -m spacy download en_core_web_sm
	pip install git+https://github.com/stanford-futuredata/ColBERT.git
	@echo "✓ Installation complete"

install-dev:
	@echo "Installing RALFS with dev dependencies..."
	pip install -e ".[dev]"
	python -m spacy download en_core_web_sm
	pip install git+https://github.com/stanford-futuredata/ColBERT.git
	pre-commit install
	@echo "✓ Dev installation complete"

# Testing
test:
	@echo "Running tests..."
	pytest tests/ -v -m "not slow"

test-fast:
	@echo "Running fast tests..."
	pytest tests/ -v -m "not slow and not integration"

test-all:
	@echo "Running all tests (including slow)..."
	pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=ralfs --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

# Linting and formatting
lint:
	@echo "Running linters..."
	ruff check ralfs/ tests/
	mypy ralfs/
	@echo "✓ Linting complete"

format:
	@echo "Formatting code..."
	black ralfs/ tests/
	isort ralfs/ tests/
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
	ralfs preprocess --dataset arxiv --max-samples 10

build-index:
	@echo "Building indexes..."
	ralfs build-index --dataset arxiv

train:
	@echo "Training model..."
	ralfs train --dataset arxiv --config configs/train/debug.yaml

evaluate:
	@echo "Running evaluation..."
	ralfs evaluate results/predictions.json data/test/references.json

# Quick start
quickstart:
	@echo "Running quick start..."
	bash scripts/quickstart.sh
