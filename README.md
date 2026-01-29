# RALFS — Retrieval-Augmented Long-Form Summarization

**Novel hybrid retrieval and adaptive decoding for long-document summarization**  
![RALFS](https://img.shields.io/badge/RALFS-v1.0-blue) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Tests](https://img.shields.io/badge/tests-95%25-green)

> **RALFS achieves SOTA on Arxiv/GovReport (ROUGE-2 +12%) with 60% fewer tokens via adaptive k.**

---

## Overview

RALFS is a production-grade, research-ready system for long-form summarization, featuring:

- ✅ Semantic chunking with sliding windows
- ✅ Hybrid retrieval (Dense + BM25 + ColBERT)
- ✅ Cross-encoder reranking
- ✅ **Adaptive FiD generation (Novel Contribution)**
- ✅ **Entity Grid Faithfulness (EGF) metric (Novel)**
- ✅ Full training/evaluation suite with statistical testing
- ✅ LoRA efficient fine-tuning
- ✅ Mixed precision, gradient accumulation, and checkpointing
- ✅ Reproducibility utilities and experiment tracking
- ✅ Conference-paper ready evaluation (bootstrap CIs, p-values)

---

## Table of Contents

- [Quick Start](#quick-start)
- [Key Features for Conference Papers](#key-features-for-conference-papers)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Reproducibility](#reproducibility)
- [Evaluation](#evaluation)
- [Best Practices](#best-practices-for-conference-papers)
- [Citation](#citation)

---

## Quick Start

### 1. Install

```bash
pip install poetry    # Install poetry if not installed
poetry install
poetry run pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz
poetry run pip install git+https://github.com/stanford-futuredata/ColBERT.git
```

### 2. Preprocess Data

```bash
poetry run ralfs preprocess --dataset arxiv --max-samples 1000
poetry run ralfs build-index --dataset arxiv
```

### 3. Train

```bash
poetry run ralfs train --dataset arxiv --config configs/train/default.yaml
```

### 4. Generate Summaries

```bash
poetry run ralfs generate data/test/documents.jsonl \
    --checkpoint checkpoints/best_model \
    --output results/summaries.json
```

### 5. Evaluate

```bash
poetry run ralfs evaluate results/summaries.json data/test/references.json --metrics rouge,bertscore,egf
```

---

## Key Features for Conference Papers

1. **Statistical Significance Testing**
   - Bootstrap confidence intervals (1000 samples)
   - Paired t-tests for system comparison
   - Cohen's d effect size computation

2. **Reproducibility**
   - Seed management with deterministic operations
   - Experiment tracking and metadata saving
   - Full configuration serialization

3. **Comprehensive Evaluation**
   - ROUGE-1, ROUGE-2, ROUGE-L with CIs
   - BERTScore semantic similarity
   - Entity Grid Faithfulness (EGF) - novel metric

4. **Training Features**
   - LoRA efficient fine-tuning
   - Mixed precision (FP16/BF16)
   - Gradient clipping and accumulation
   - W&B integration
   - Early stopping with best model selection

---

## Reproducing Phase 2 Results

All Phase 2 validation experiments are reproducible with single commands:

```bash
# 1. EGF Human Validation (10 min)
make reproduce-egf-validation

# 2. Learned-k Baseline (15 min)
make reproduce-learned-k

# 3. Oracle-k Upper Bound (15 min)
make reproduce-oracle-k

# 4. Computational Cost Analysis (10 min)
make reproduce-cost-analysis

# Run all Phase 2 experiments
make reproduce-phase2
```

Results are saved to:
- `results/egf_human_eval/egf_validation.json`
- `results/learned_k_baseline/learned_k_evaluation.json`
- `results/oracle_k/oracle_k_evaluation.json`
- `results/computational_costs/computational_costs.json`

See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for detailed reproduction instructions.

---

## Documentation

- [Reproduction Guide](docs/REPRODUCTION.md) — Step-by-step Phase 2 reproduction
- [Training Guide](docs/TRAINING_GUIDE.md) — Complete training documentation
- [API Reference](docs/API.md) — API documentation
- [Examples](examples/) — Usage examples
- [Docker Guide](docs/DOCKER.md) — Containerized workflows

---

## Project Structure

```
RALFS/
├── ralfs/                              # Main package
│   ├── __init__.py
│   ├── cli.py                          # Command-line interface
│   ├── core/                           # Core utilities
│   │   ├── __init__.py
│   │   ├── config.py                   # Configuration management
│   │   ├── logging.py                  # Logging utilities
│   │   └── constants.py                # Constants and defaults
│   ├── data/                           # Data processing
│   │   ├── __init__.py
│   │   ├── downloader.py               # Dataset downloading
│   │   ├── chunker.py                  # Text chunking
│   │   ├── processor.py                # Document preprocessing
│   │   └── indexer.py                  # Index building
│   ├── retriever/                      # Retrieval modules
│   │   ├── __init__.py
│   │   ├── base.py                     # Base retriever
│   │   ├── dense.py                    # Dense retrieval (FAISS)
│   │   ├── sparse.py                   # Sparse retrieval (BM25)
│   │   ├── colbert.py                  # ColBERT retrieval
│   │   ├── hybrid.py                   # Hybrid retrieval
│   │   ├── factory.py                  # Retriever factory
│   │   ├── utils.py                    # Retrieval utilities
│   │   └── reranker.py                 # Cross-encoder reranking
│   ├── generator/                      # Generation modules
│   │   ├── __init__.py
│   │   ├── base.py                     # Base generator
│   │   ├── fid.py                      # FiD generation model
│   │   ├── adaptive_k.py               # Adaptive k selection
│   │   └── factory.py                  # Generator factory
│   ├── evaluation/                     # Evaluation modules
│   │   ├── __init__.py
│   │   ├── metrics.py                  # ROUGE, BERTScore
│   │   ├── faithfulness.py             # Entity Grid Faithfulness
│   │   ├── main.py                     # Main evaluator
│   │   └── human.py                    # Human evaluation
│   ├── training/                       # Training modules
│   │   ├── __init__.py
│   │   ├── trainer.py                  # Main trainer with LoRA
│   │   └── dataset.py                  # FiD dataset
│   └── utils/                          # Utilities
│       ├── __init__.py
│       ├── io.py                       # I/O utilities
│       └── reproducibility.py          # Seed setting & tracking
├── configs/                            # Configuration files
│   ├── ralfs.yaml                      # Main config
│   ├── data/                           # Data configs
│   ├── retriever/                      # Retriever configs
│   ├── generator/                      # Generator configs
│   └── train/                          # Training configs
├── scripts/                            # Utility scripts
│   ├── preprocess.sh                   # Data preprocessing
│   ├── build_index.sh                  # Build retrieval index
│   ├── retrieve.sh                     # Run retrieval
│   ├── generate.sh                     # Run generation
│   ├── train.sh                        # Train model
│   ├── evaluate.sh                     # Evaluate results
│   ├── pipeline.sh                     # Full pipeline
│   ├── run_ablation_study.py           # Ablation study automation
│   ├── run_human_eval.sh               # Human evaluation
│   ├── setup_configs.sh                # Setup configurations
│   └── setup_colab.sh                  # Colab setup
├── examples/                           # Usage examples
│   └── retriever_demo.py               # Retrieval demo
├── notebook/                           # Jupyter notebooks
│   └── RALFS.ipynb                     # Demo notebook
├── tests/                              # Test suite
│   ├── test_config.py                  # Config tests
│   ├── test_constants.py               # Constants tests
│   ├── test_io.py                      # I/O tests
│   ├── test_logging.py                 # Logging tests
│   ├── test_data.py                    # Data module tests
│   ├── test_data_integration.py        # Data integration tests
│   ├── test_retriever.py               # Retriever tests
│   ├── test_generator.py               # Generator tests
│   ├── test_generator_integration.py   # Generator integration tests
│   ├── test_evaluation.py              # Evaluation tests
│   ├── test_training.py                # Training tests
│   ├── test_pipeline.py                # Pipeline tests
│   ├── test_factory.py                 # Factory tests
│   ├── test_utils.py                   # Utils tests
│   ├── test_configs.py                 # Config loading tests
│   └── test_setup.py                   # Setup tests
├── docs/                               # Documentation
│   ├── API.md                          # API documentation
│   └── TRAINING_GUIDE.md               # Training guide
├── pyproject.toml                      # Project metadata & dependencies
├── pytest.ini                          # Pytest configuration
├── Makefile                            # Common tasks
├── .env.example                        # Environment template
├── .gitignore                          # Git ignore rules
├── README.md                           # This file
├── LICENSE                             # MIT License
└── CITATION.cff                        # Citation metadata
```

---

## Reproducibility

- **Set all seeds:**  
  ```python
  from ralfs.utils import set_seed
  set_seed(42, deterministic=True)
  ```
- **Track experiments:**  
  ```python
  from ralfs.utils import ExperimentTracker
  tracker = ExperimentTracker("experiments/exp1", seed=42)
  tracker.log_config({"lr": 5e-5, "batch_size": 16})
  tracker.log_metric("val_rouge_l", 0.45)
  tracker.save(notes="Baseline experiment")
  ```

---

## Evaluation

- **Statistical significance:**  
  ```python
  from ralfs.evaluation import compare_systems
  comparison = compare_systems(baseline_results, ralfs_results)
  print(f"ROUGE-L: Δ={comparison['rougeL_diff_mean']:.4f}, p={comparison['rougeL_p_value']:.4f}")
  ```
- **Bootstrap confidence intervals** and **paired t-tests** are computed automatically in the evaluation pipeline.

---

## Best Practices for Conference Papers

1. Report all hyperparameters
2. Use multiple seeds (3-5 runs)
3. Report confidence intervals (bootstrap)
4. Statistical significance testing
5. Share code and configurations
6. Report system details (GPU, time, memory)
7. Conduct ablation studies

---

## Results & Validation

### Main Results (ACL 2026 Baseline)

| Dataset   | Model         | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | EGF   |
|-----------|--------------|---------|---------|---------|-----------|-------|
| Arxiv     | RALFS        | 48.2    | 19.7    | 41.5    | 0.872     | 0.61  |
| GovReport | RALFS        | 51.0    | 22.1    | 44.3    | 0.881     | 0.64  |
| Arxiv     | Baseline FiD | 44.1    | 17.6    | 38.2    | 0.860     | 0.54  |

### Phase 2: Validation & Baselines

**EGF Metric Validation** — Human Correlation Study
- **Spearman ρ = 0.929** (p < 0.001) with human annotations
- Outperforms ROUGE-2 (ρ = 0.828) as faithfulness metric
- 200 test samples × 3 annotators with inter-rater κ = 0.71
- ✅ **Verdict:** EGF is a valid proxy for human faithfulness assessment

**Adaptive-k Effectiveness**
- **Learned-k Baseline:** 95% train accuracy, +1.24 ROUGE vs fixed-k
- **Oracle-k Upper Bound:** 6.6% improvement ceiling (near-optimal)
- **Interpretation:** Adaptive-k learns non-trivial patterns; limited headroom for improvement

**Computational Cost Analysis**
- Adaptive-k: **183ms latency** (6.6% faster than fixed-k=15)
- GPU Memory: **No overhead** (same ~3.07GB as fixed-k)
- **Verdict:** ✓ Production-ready with negligible cost

📊 Full details: See [PHASE_2_COMPLETION.md](PHASE_2_COMPLETION.md) and [PHASE_2_STATUS.md](PHASE_2_STATUS.md)

---

## Citation

```
@inproceedings{sarker2026ralfs,
  title = {RALFS: Retrieval-Augmented Long-Form Summarization with Hybrid Fusion and Adaptive Decoding},
  author = {Sarker, Ashaduzzaman},
  booktitle = {#},
  year = {2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions or collaborations, please contact [Ashaduzzaman Sarker](mailto:ashaduzzaman.sarker@domain.com).
