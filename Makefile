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
	@echo "  install           - Install dependencies"
	@echo "  test              - Run tests"
	@echo "  test-fast         - Run fast tests only"
	@echo "  test-all          - Run all tests including slow ones"
	@echo "  lint              - Run linters"
	@echo "  format            - Format code"
	@echo "  clean             - Clean generated files"
	@echo "  preprocess        - Preprocess data"
	@echo "  build-index       - Build retrieval indexes"
	@echo "  train             - Train model"
	@echo "  evaluate          - Run evaluation"
	@echo ""
	@echo "Reproduction targets:"
	@echo "  reproduce-paper      - Reproduce all paper results (main + ablations)"
	@echo "  reproduce-table1     - Reproduce Table 1: Main Results"
	@echo "  reproduce-table2     - Reproduce Table 2: Ablations"
	@echo "  reproduce-phase2     - Reproduce all Phase 2 validation (EGF+baselines+costs)"
	@echo "  reproduce-learned-k         - Phase 2.1: Learned-k baseline"
	@echo "  reproduce-oracle-k          - Phase 2.2: Oracle-k upper bound"
	@echo "  reproduce-egf-validation    - Phase 2.3: EGF human validation"
	@echo "  reproduce-cost-analysis     - Phase 2.4: Computational cost analysis"
	@echo "  reproduce-retrieval-ablation - Phase 2.5: Retrieval ablation study"

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

# Reviewer targets
reviewer-test:
	@echo "Running reviewer smoke test (< 5 min)..."
	poetry run pytest tests/test_reviewer_smoke.py -v -s
	@echo "✓ Smoke test complete"

reproduce-paper:
	@echo "Reproducing all paper results (Table 1 + Table 2 + ablations)..."
	@echo "⚠️  WARNING: This will take ~16 hours on A100"
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	poetry run python -m experiments.table1_main_results --datasets arxiv govreport --seeds 13 42 2024
	poetry run python -m experiments.table2_ablations --seed 42
	@echo "✓ Paper reproduction complete"
	@echo "Results in: results/main_results/ and results/ablations/"

reproduce-table1:
	@echo "Reproducing Table 1: Main Results..."
	poetry run python -m experiments.table1_main_results --datasets arxiv govreport --seeds 13 42 2024
	@echo "✓ Table 1 complete: results/main_results/table1.csv"

reproduce-table2:
	@echo "Reproducing Table 2: Ablations..."
	poetry run python -m experiments.table2_ablations --seed 42
	@echo "✓ Table 2 complete: results/ablations/table2.csv"

# Phase 2 Validation & Baselines
reproduce-egf-validation:
	@echo "Phase 2.3: EGF Human Validation Study..."
	@echo "⏱️  Estimated time: 10 minutes"
	poetry run python experiments/egf_human_eval.py
	@echo "✓ EGF validation complete: results/egf_human_eval/egf_validation.json"

reproduce-learned-k:
	@echo "Phase 2.1: Learned-k Baseline..."
	@echo "⏱️  Estimated time: 15 minutes"
	poetry run python experiments/baselines/learned_k.py
	@echo "✓ Learned-k baseline complete: results/learned_k_baseline/learned_k_evaluation.json"

reproduce-oracle-k:
	@echo "Phase 2.2: Oracle-k Upper Bound..."
	@echo "⏱️  Estimated time: 12 minutes"
	poetry run python experiments/baselines/oracle_k.py
	@echo "✓ Oracle-k bound complete: results/oracle_k/oracle_k_evaluation.json"

reproduce-cost-analysis:
	@echo "Phase 2.4: Computational Cost Analysis..."
	@echo "⏱️  Estimated time: 10 minutes"
	poetry run python experiments/compute_costs.py
	@echo "✓ Cost analysis complete: results/computational_costs/computational_costs.json"

reproduce-retrieval-ablation:
	@echo "Phase 2.5: Retrieval Ablation Study..."
	@echo "⏱️  Estimated time: 5 minutes"
	python3 experiments/retrieval_ablation.py --num-queries 200 --seed 42
	@echo "✓ Retrieval ablation complete: results/retrieval_ablation/retrieval_ablation.json"

reproduce-phase2: reproduce-egf-validation reproduce-learned-k reproduce-oracle-k reproduce-cost-analysis reproduce-retrieval-ablation
	@echo ""
	@echo "╔════════════════════════════════════════════╗"
	@echo "║  ✓ Phase 2 Reproduction Complete (5/5)    ║"
	@echo "╚════════════════════════════════════════════╝"
	@echo ""
	@echo "Results saved to:"
	@echo "  • results/egf_human_eval/egf_validation.json"
	@echo "  • results/learned_k_baseline/learned_k_evaluation.json"
	@echo "  • results/oracle_k/oracle_k_evaluation.json"
	@echo "  • results/computational_costs/computational_costs.json"
	@echo "  • results/retrieval_ablation/retrieval_ablation.json"
	@echo ""
	@echo "See docs/REPRODUCTION.md for detailed results and interpretation."

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
