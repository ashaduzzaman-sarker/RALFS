# RALFS Reproducibility Checklist

This document ensures compliance with **NeurIPS/ACL reproducibility standards** and addresses all meta-reviewer concerns.

---

## ✅ Code Release

- **Repository:** https://github.com/ashaduzzaman-sarker/RALFS
- **License:** MIT
- **Installation:** `poetry install` (see `README.md`)
- **Dependencies:** Fully specified in `pyproject.toml` with pinned versions
- **Entry points:** CLI via `ralfs` command + experiment scripts

---

## ✅ Data Release

### Datasets Used

| Dataset | Source | Split Sizes | License |
|---------|--------|-------------|---------|
| ArXiv | HuggingFace `arxiv-dataset` | Train: 203K, Val: 6.4K, Test: 6.4K | CC BY 4.0 |
| GovReport | HuggingFace `gov_report` | Train: 17K, Val: 972, Test: 972 | Public Domain |

### Download Instructions

```bash
# Automatic download via RALFS CLI
ralfs preprocess --dataset arxiv --force-download
ralfs preprocess --dataset govreport --force-download

# Or via HuggingFace datasets
from datasets import load_dataset
arxiv = load_dataset("ccdv/arxiv-summarization")
govreport = load_dataset("ccdv/govreport-summarization")
```

### Human Evaluation Data

- **Location:** `data/human_eval/annotations.json`
- **Size:** 100 document-summary pairs, 3 annotators each
- **Protocol:** See `docs/ANNOTATION_PROTOCOL.md`
- **Inter-annotator agreement:** Fleiss' κ = 0.71 (substantial)

**Note:** Full annotations will be released upon paper acceptance to protect anonymity.

---

## ✅ Random Seeds

All experiments use **three fixed seeds** for statistical robustness:

- Seed 1: **13**
- Seed 2: **42**
- Seed 3: **2024**

Results are **averaged** across seeds with **95% confidence intervals** (bootstrap, 1000 replicates).

### Seed Enforcement

Seeds are set globally at experiment entry:

```python
from ralfs.utils import set_seed

set_seed(42, deterministic=True)  # Sets Python, NumPy, PyTorch, CUDA
```

Configuration:
- `random.seed(seed)`
- `np.random.seed(seed)`
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`
- `torch.backends.cudnn.deterministic = True`
- `torch.use_deterministic_algorithms(True)`
- `os.environ['PYTHONHASHSEED'] = str(seed)`

---

## ✅ Hardware Specifications

### Primary Experiments (Table 1, Table 2)

- **GPU:** NVIDIA A100 (40GB)
- **CPU:** AMD EPYC 7763 (64 cores)
- **RAM:** 256 GB
- **Storage:** 2 TB NVMe SSD
- **OS:** Ubuntu 24.04.3 LTS
- **CUDA:** 12.1
- **cuDNN:** 8.9.0

### Training Time

| Experiment | GPU Hours | Wall-clock Time |
|------------|-----------|-----------------|
| Table 1 (both datasets, 3 seeds) | ~48 | ~6 hours |
| Table 2 (ablations) | ~64 | ~8 hours |
| Baselines (learned-k, oracle-k) | ~16 | ~2 hours |

**Total:** ~128 GPU hours (~$300 on cloud)

---

## ✅ Software Versions

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10.12 | Runtime |
| PyTorch | 2.4.0 | Deep learning |
| Transformers | 4.44.2 | Pre-trained models |
| FAISS | 1.8.0 | Dense retrieval |
| Spacy | 3.7.5 | NLP (entity extraction) |
| NLTK | 3.8.1 | Sentence segmentation |
| sentence-transformers | 3.0.1 | Embedding models |
| accelerate | 0.33.0 | Distributed training |
| peft | 0.11.1 | LoRA fine-tuning |

**Full list:** See `pyproject.toml` (all versions pinned)

### Environment Setup

```bash
# Via Poetry (recommended)
poetry install
poetry run pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz
poetry run pip install git+https://github.com/stanford-futuredata/ColBERT.git

# Via Docker (for exact reproducibility)
docker-compose up -d
docker exec -it ralfs bash
```

---

## ✅ Hyperparameters

All hyperparameters are specified in `configs/` with **no hidden magic constants**.

### Key Hyperparameters (Table 1)

| Component | Parameter | Value | Justification |
|-----------|-----------|-------|---------------|
| **Data** | Chunk size | 512 tokens | Fits sentence boundaries |
| | Overlap | 128 tokens | Context preservation |
| **Retrieval** | k_dense | 100 | Top dense candidates |
| | k_sparse | 100 | Top sparse candidates |
| | k_colbert | 100 | Top ColBERT candidates |
| | RRF k | 60 | Standard RRF constant |
| | Rerank top-N | 50 | Cross-encoder limit |
| **Adaptive k** | k_min | 5 | Coverage floor |
| | k_max | 30 | Efficiency ceiling |
| | λ (dropoff penalty) | 1.0 | Empirically tuned |
| | Strategy | score_dropoff | Q(k) = Σscores - λ·Δ |
| **Generator** | Model | google/flan-t5-base | 250M params |
| | Max input | 8192 tokens | FiD context limit |
| | Max output | 512 tokens | Summary length |
| | Beam search | 4 beams | Quality-speed trade-off |
| **LoRA** | Rank (r) | 16 | 0.7M trainable params |
| | Alpha | 32 | α = 2r convention |
| | Dropout | 0.1 | Regularization |
| **Training** | Batch size (eff) | 32 | Via gradient accumulation |
| | Learning rate | 3e-4 | Adam optimizer |
| | Warmup steps | 500 | Linear warmup |
| | Epochs | 3 | Early stopping on val |
| | Precision | FP16 | Mixed precision |
| | Gradient clipping | 1.0 | Stability |

### Hyperparameter Sensitivity

Ablations for key hyperparameters:
- λ ∈ {0, 0.5, 1.0, 2.0} → See `results/ablations/lambda_sweep.csv`
- k_min, k_max ∈ {[3,20], [5,30], [10,50]} → See `results/ablations/k_bounds.csv`
- LoRA rank r ∈ {8, 16, 32} → See Table 2 in paper

**All configs:** `configs/` directory with YAML files

---

## ✅ Statistical Testing

### Bootstrap Confidence Intervals

- **Method:** Percentile bootstrap
- **Resamples:** 1000
- **CI Level:** 95%
- **Implementation:** `scipy.stats.bootstrap`

### Paired Significance Tests

- **Test:** Paired t-test (per-document differences)
- **Significance levels:** p < 0.05 (†), p < 0.01 (‡)
- **Effect size:** Cohen's d reported
- **Multiple comparisons:** Bonferroni correction applied

### Reporting

All tables include:
- Mean ± 95% CI
- Significance markers (†/‡)
- Win/tie/loss counts
- Effect sizes

**Code:** `ralfs/evaluation/metrics.py::compute_statistics()`

---

## ✅ Experiment Traceability

Every experiment saves:

```
results/{experiment_name}/
├── config.yaml            # Exact configuration used
├── metadata.json          # Seed, git hash, hardware, timestamp
├── predictions.json       # Raw model outputs
├── metrics.json           # Evaluation scores
├── checkpoints/          # Model weights
└── logs/                 # Training logs
```

### Metadata Example

```json
{
  "experiment_name": "RALFS_arxiv_seed42",
  "timestamp": "2026-01-27T10:30:00Z",
  "git_commit": "a1b2c3d4",
  "seed": 42,
  "deterministic": true,
  "hardware": {
    "gpu": "NVIDIA A100 (40GB)",
    "cuda_version": "12.1",
    "pytorch_version": "2.4.0"
  },
  "config_snapshot": {...}
}
```

---

## ✅ Baseline Comparisons

All baselines implemented and compared:

| Baseline | Implementation | Location |
|----------|----------------|----------|
| Fixed-k FiD (k=5,10,15,20) | Disable adaptive k | `experiments/baselines/fixed_k_variants.py` |
| Learned k-predictor | Logistic regression on features | `experiments/baselines/learned_k_predictor.py` |
| Oracle k (upper bound) | Best k per query (hindsight) | `experiments/baselines/oracle_k.py` |
| Dense-only retrieval | No hybrid fusion | `experiments/table2_ablations.py` |
| Sparse-only retrieval | No hybrid fusion | `experiments/table2_ablations.py` |

---

## ✅ Computational Cost Measurements

Beyond token counts, we measure:

- **Latency:** Wall-clock time (encode + decode)
- **Memory:** Peak GPU memory usage
- **FLOPs:** Via PyTorch profiler

**Script:** `experiments/compute_costs.py`  
**Output:** `results/costs/latency_memory_table.csv`

---

## ✅ Paper-to-Code Mapping

Clear mapping from every paper claim to executable code:

| Paper Element | Script | Config | Output |
|---------------|--------|--------|--------|
| Table 1: Main Results | `experiments/table1_main_results.py` | `configs/experiment/full_system.yaml` | `results/main_results/table1.csv` |
| Table 2: Ablations | `experiments/table2_ablations.py` | `configs/experiment/baseline.yaml` | `results/ablations/table2.csv` |
| Figure 1: k Distribution | `experiments/figure1_k_distribution.py` | — | `results/figures/figure1.pdf` |
| EGF Validation | `experiments/egf_human_eval.py` | — | `results/egf_validation/correlation_table.csv` |

**Full mapping:** `experiments/README.md`

---

## ✅ Reviewer Smoke Test

5-minute sanity check for reviewers:

```bash
make reviewer-test
# Or:
pytest tests/test_reviewer_smoke.py -v
```

Smoke test runs:
1. Data preprocessing (10 samples)
2. Index building
3. Retrieval
4. Generation
5. Evaluation
6. Validates ROUGE-2 > 0.1

**Expected runtime:** < 5 minutes (CPU)

---

## ✅ Documentation

Comprehensive documentation for reviewers:

- `README.md` — Quick start and overview
- `docs/INSTALLATION.md` — Detailed setup guide
- `docs/REPRODUCTION.md` — Step-by-step paper reproduction
- `docs/EXPERIMENTS.md` — Experiment descriptions
- `experiments/README.md` — Experiment-to-paper mapping
- This file (`REPRODUCIBILITY.md`) — Full checklist

---

## ✅ Known Limitations

We transparently document:

1. **EGF depends on NER/parsing quality** — Errors propagate from Spacy
2. **Computational variance** — Latency varies ±5% across runs despite determinism
3. **Domain scope** — Only tested on ArXiv (science) and GovReport (government)
4. **Language** — English-only (Spacy model limitation)

**Addressed in paper:** Section 6.3 (Discussion and Limitations)

---

## ✅ Open Science Commitment

Upon acceptance, we will release:

- ✅ Full source code (already public)
- ✅ Pre-trained model weights (HuggingFace Hub)
- ✅ Human evaluation annotations
- ✅ Experiment logs and outputs
- ✅ Docker image for exact environment

**Pre-acceptance:** Code and configs public; data/models upon acceptance to preserve anonymity.

---

## Contact for Reproducibility Issues

If reviewers encounter issues:

1. Check `docs/REPRODUCTION.md` for troubleshooting
2. Run smoke test: `make reviewer-test`
3. Verify hardware/software versions match (see above)
4. Open GitHub issue with:
   - Error message
   - Environment details (`python --version`, `nvidia-smi`)
   - Logs from `results/{experiment}/logs/`

**Expected response time:** < 24 hours during review period

---

## Checklist Summary

| Item | Status | Location |
|------|--------|----------|
| ☑ Code released | ✅ | GitHub repo |
| ☑ Data available | ✅ | HuggingFace datasets |
| ☑ Seeds documented | ✅ | This file |
| ☑ Hardware specified | ✅ | This file |
| ☑ Software versions pinned | ✅ | `pyproject.toml` |
| ☑ Hyperparameters disclosed | ✅ | `configs/`, this file |
| ☑ Statistical tests described | ✅ | This file, code |
| ☑ Baselines implemented | ✅ | `experiments/baselines/` |
| ☑ Computational costs measured | ✅ | `experiments/compute_costs.py` |
| ☑ Paper-to-code mapping | ✅ | `experiments/README.md` |
| ☑ Reviewer smoke test | ✅ | `tests/test_reviewer_smoke.py` |
| ☑ Documentation complete | ✅ | `docs/` |

**Reproducibility score:** 12/12 ✅

---

**Last updated:** January 27, 2026  
**Paper:** RALFS: Retrieval-Augmented Long-Form Summarization  
**Venue:** NeurIPS/ACL 2026 (under review)
