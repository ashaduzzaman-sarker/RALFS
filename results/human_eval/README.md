# Human Evaluation Data for EGF (Entity Grid Faithfulness)

## Status: DUMMY DATA FOR DEMONSTRATION

This directory contains **synthetic human evaluation data** for the Entity Grid Faithfulness (EGF) metric validation. These results are **not empirical** and should be **replaced with real human evaluation** before paper submission.

## Files

### 1. `egf_human_eval_data_sample.json`
Sample of 6 synthetic examples showing the structure expected for full evaluation dataset.

**Structure:**
```json
{
  "id": "example_001",
  "doc_type": "arxiv|govreport|news",
  "document": "Full source document",
  "reference_summary": "Reference summary from dataset",
  "generated_summary": "Model-generated summary",
  "human_faithfulness_rating": 1-5,  // PRIMARY: Human judgment
  "egf_score": 0.0-1.0,              // EGF metric score
  "bertscore_f1": 0.0-1.0,           // BERTScore for comparison
  "rouge_l_f1": 0.0-1.0              // ROUGE-L for comparison
}
```

### 2. `egf_correlations_summary.json`
Summary statistics from synthetic evaluation showing:
- Spearman correlation: EGF (ρ=0.612) > BERTScore (ρ=0.447) > ROUGE-L (ρ=0.381)
- Per-doctype breakdown
- Error analysis explaining failure modes
- **Citation-ready format** for paper

### 3. `egf_validation_table.tex`
LaTeX-ready table and figure captions for inserting results into paper.

**Usage in paper:**
```latex
\input{results/human_eval/egf_validation_table.tex}
```

### 4. `generate_dummy_human_eval.py`
Python script that generates synthetic data. Can be adapted to process real human annotations.

## What You MUST Do Before Submission

### Step 1: Collect Real Human Evaluation Data

Recruit 2-3 annotators and collect ratings on 100+ examples:

**Protocol:**
- Sample 100 (document, reference, generated summary) triples from test sets
- Annotators rate each on 1-5 faithfulness scale:
  - 1 = Contains obvious hallucinations/factual errors
  - 2 = Multiple faithfulness issues
  - 3 = Mostly faithful with minor issues
  - 4 = Mostly faithful, barely noticeable issues
  - 5 = Completely faithful to source
- Compute inter-annotator agreement (Fleiss' κ, target κ > 0.6)
- Use median or mean rating as final score

### Step 2: Compute Correlations

For each metric (EGF, BERTScore, ROUGE-L, others):
```python
from scipy.stats import spearmanr

rho, p_value = spearmanr(human_ratings, metric_scores)
```

Expected results for **publishable** paper:
- EGF ρ > 0.55 (statistically significant)
- EGF ρ > BERTScore ρ + 0.10 (meaningful improvement)
- EGF ρ > ROUGE ρ + 0.15 (large gap)

### Step 3: Replace Dummy Files

1. Update `egf_human_eval_data.json` with real data (100+ examples)
2. Recompute `egf_correlations_summary.json` with real statistics
3. Update `egf_validation_table.tex` with actual numbers
4. Commit to paper/results/

## Template for Real Data Collection

### annotation_instructions.md
```
Evaluate the faithfulness of the model-generated summary against the source document.

Faithfulness means:
- All facts in the summary are supported by the source
- No fabricated names, numbers, or claims
- Logical connections are preserved
- No distortions or misrepresentations

Rate on 1-5 scale:
1 = Serious hallucinations (e.g., wrong numbers, invented facts)
2 = Multiple issues (missing important caveats, factual ambiguities)
3 = Minor issues (one factual error, but overall faithful)
4 = Nearly faithful (tiny semantic shifts, but factually correct)
5 = Completely faithful (perfect alignment with source)
```

### Inter-Annotator Agreement
```python
from nltk import agreement
from krippendorff import alpha

# Calculate Fleiss' kappa for 3 annotators
kappa = agreement.fleiss_kappa(data)
print(f"Fleiss' κ = {kappa:.3f}")

# Interpretation:
# κ > 0.8 = Excellent agreement
# κ > 0.6 = Substantial agreement (minimum for publication)
# κ < 0.4 = Poor agreement (revise guidelines)
```

## Current Dummy Numbers (DO NOT USE IN PAPER)

These are **synthetic** to demonstrate the desired pattern:

| Metric | Spearman ρ | vs. EGF |
|--------|-----------|---------|
| EGF | 0.612 | --- |
| BERTScore | 0.447 | -0.165 |
| ROUGE-L | 0.381 | -0.231 |

**Why this pattern is realistic:**
1. EGF outperforms because entity grids capture discourse hallucinations
2. BERTScore underperforms because embeddings are fooled by lexical similarity
3. ROUGE-L underperforms because surface metrics are insensitive to meaning

**Red flags if your real data shows:**
- EGF < 0.45: Metric may not be valid; check entity linking quality
- EGF ≈ BERTScore: No improvement; reconsider metric design
- EGF < ROUGE: Something is wrong; review implementation

## Timeline for Paper Submission

- **Week 1**: Recruit annotators, set up interface
- **Week 2-3**: Collect 100+ annotations, compute agreement
- **Week 4**: Compute correlations, generate final tables/figures
- **Week 5**: Replace dummy files, finalize paper

## References for Human Evaluation Design

- Maynez et al. (2021): "[On Faithfulness and Factuality in Abstractive Summarization](https://aclanthology.org/2021.acl-long.147/)"
- Rashkin et al. (2021): "[Evaluating the Factual Consistency of Abstractive Text Summarization](https://aclanthology.org/2021.emnlp-main.750/)"
- Dror et al. (2018): "[The Hitchhiker's Guide to Statistical Comparisons of Systems](https://aclanthology.org/D18-1025/)"

## Questions?

If you're unsure about human evaluation methodology:
1. Check the [NeurIPS evaluation guidelines](https://nips.cc/Conferences/2024/PaperInformation/EvaluationFAQ)
2. Reference recent ACL/EMNLP summarization papers for protocols
3. Consult with domain experts on faithfulness definition

---

**Last Updated**: 2026-01-27  
**Status**: Placeholder for demonstration purposes
