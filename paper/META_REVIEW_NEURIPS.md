# NeurIPS Meta-Review: RALFS

**Paper ID:** [Redacted]  
**Title:** RALFS: Retrieval-Augmented Long-Form Summarization with Adaptive Passage Selection and Entity Grid Faithfulness

**Meta-Reviewer:** [Anonymous]  
**Date:** January 27, 2026

---

## Review Summary

The paper received one detailed review scoring 5/10 (Reject). The reviewer acknowledges strong reproducibility practices (8/10) and clear writing (7/10) but raises substantive concerns about **novelty** (4/10), **technical quality** (5/10), and **significance** (5/10).

**Reviewer's main criticisms:**
1. **Incremental novelty**: Adaptive k selection is a straightforward heuristic (Q(k) = cumulative score - λ·dropoff) without theoretical grounding
2. **Missing critical baselines**: No comparison to learned k-predictors, oracle k, or confidence thresholds
3. **Insufficient EGF validation**: Human evaluation claim (ρ = 0.612) lacks supporting evidence (no table, protocol, or inter-annotator agreement)
4. **Limited evaluation breadth**: Only 2 datasets (ArXiv, GovReport); no error analysis, computational cost measurements, or hybrid retrieval ablation breakdown
5. **Overstated claims**: "Principled framework" language doesn't match the heuristic nature of the method

**Reviewer's strengths noted:**
- Clear problem formulation and motivation
- Simple, deployable method
- Excellent reproducibility (versioned configs, seeds, CIs)
- Honest discussion of limitations

---

## Meta-Reviewer Assessment

### Overall Evaluation

After careful consideration, I **agree with the reviewer's concerns** but find the assessment slightly harsh given the paper's practical contributions and rigor. The core issue is that **the paper straddles the boundary between solid engineering and research novelty**—it solves a real problem with a simple, effective solution, but the scientific contribution is limited.

**Key observations:**

1. **The adaptive k contribution is incremental but useful.** The Q(k) formula is indeed heuristic, but the paper clearly differentiates it from cascade retrieval (Section 2.2.1) and demonstrates practical value. The +12% ROUGE-2 gain is non-trivial. However, the reviewer is correct that missing baselines (learned k, oracle k) make it impossible to assess whether the *specific* formula matters or just "not using fixed k."

2. **EGF validation is the paper's Achilles' heel.** The claim of superior human correlation (ρ = 0.612 vs. 0.447) is central to the contribution but **completely unsubstantiated**. The referenced Table~\ref{tab:human_eval} doesn't exist. This is a **fatal flaw** that must be addressed. Without this evidence, EGF is speculative.

3. **Experimental rigor is strong but incomplete.** The reproducibility effort (Table 1, configs, seeds) is commendable and above average. However, critical ablations are missing:
   - No breakdown of hybrid retrieval components (which drives the "sharper score distributions")
   - No error analysis (k selection distribution, failure modes)
   - No computational cost beyond token counts

4. **Limited scope affects significance.** Two datasets from similar domains (formal, long documents) is thin for a NeurIPS submission. The generalization claims in Discussion are speculative without evidence on news, dialogue, or multilingual data.

5. **Writing quality vs. precision.** The paper is well-written, but language like "principled framework" and "optimization problem" overstates the theoretical grounding. The method is a practical heuristic, which is fine—but the framing should match.

### Comparison to NeurIPS Standards

NeurIPS typically accepts papers with:
- **Strong novelty** (new algorithms, theory, or phenomena) OR
- **Strong empirical contributions** (comprehensive evaluation, surprising findings, clear practical impact)

This paper has:
- ✓ Clear practical motivation
- ✓ Excellent reproducibility
- ✗ Limited novelty (straightforward heuristic)
- ✗ Incomplete evaluation (missing baselines, limited datasets)
- ✗ One major contribution (EGF) is unvalidated

**Verdict: The paper falls below the NeurIPS bar in its current form.**

---

## Decision

**BORDERLINE REJECT (Score: 5.5/10)**

**Justification:**

The paper addresses a real problem with a practical solution and demonstrates strong experimental rigor in reproducibility. However, **three critical issues prevent acceptance**:

1. **Missing EGF validation** (blocking issue)
2. **Insufficient baselines** to justify adaptive k over simpler alternatives
3. **Limited evaluation breadth** (2 datasets, no cost measurements, no error analysis)

The positive aspects (reproducibility, clear writing, practical utility) are notable but insufficient to overcome these gaps for a venue like NeurIPS that prioritizes novelty and comprehensive evaluation.

---

## Path to Acceptance

I outline **two scenarios** for the authors to reach acceptance, either at NeurIPS (major revision) or a more suitable venue.

### Scenario A: Major Revision for NeurIPS Resubmission

**Required (blocking issues):**

1. **EGF Human Evaluation [CRITICAL]**
   - Recruit ≥3 annotators (ideally domain experts)
   - Annotate ≥100 document-summary pairs on 5-point faithfulness scale
   - Report: inter-annotator agreement (Fleiss' κ or Krippendorff's α), Spearman correlation with EGF/BERTScore/ROUGE, significance tests (bootstrap or permutation)
   - Provide annotation protocol in appendix
   - **Without this, EGF is not a validated contribution**

2. **Strong Baselines for Adaptive k**
   - **Learned k-predictor**: Train logistic regression or small MLP on retrieval features (mean score, variance, top-k gap, document length) to predict k. Compare to Q(k) formula.
   - **Oracle k**: For each dev query, select k that maximizes dev ROUGE. Report this upper bound.
   - **Confidence threshold**: Simple baseline: select passages until score < threshold.
   - **Ablation**: Show performance for λ ∈ {0, 0.5, 1.0, 2.0} and k_min/k_max ranges.
   - **Goal**: Demonstrate that Q(k) formula is necessary, not just "adaptive k in general."

3. **Hybrid Retrieval Ablation**
   - Table showing: Dense-only, Sparse-only, ColBERT-only, Dense+Sparse, Dense+ColBERT, Sparse+ColBERT, All-three
   - For each: mean k selected, ROUGE, tokens
   - **Goal**: Separate contribution of fusion vs. adaptive k

**Strongly Recommended:**

4. **Error Analysis**
   - Distribution of k selected (histogram)
   - Failure case analysis: queries where k_min or k_max was selected, queries where RALFS underperformed fixed-k
   - Correlation between k and document length, genre, query difficulty

5. **Computational Cost**
   - Wall-clock latency (encoding + decoding) for adaptive k vs. fixed k (with matched avg k)
   - GPU memory usage
   - FLOPs or GPU-hours

6. **Broader Evaluation**
   - Add ≥2 more datasets: CNN/DailyMail (news), XSum (extreme summarization), or Multi-News (multi-document)
   - If not feasible, at least show per-document-length analysis on ArXiv/GovReport

**If these are addressed, the paper would be competitive for NeurIPS with scores ~6.5-7/10.**

---

### Scenario B: Target a More Suitable Venue (Recommended)

Given the applied nature and engineering focus, I recommend **ACL or EMNLP** as a better fit. These venues value:
- Practical systems with strong engineering
- Reproducibility and clear writing
- Applied contributions to NLP tasks

**For ACL/EMNLP acceptance, the authors would still need:**

**Essential:**
1. EGF human evaluation (as above)
2. At least one strong adaptive-k baseline (learned k-predictor OR oracle k)
3. Hybrid retrieval ablation

**Recommended (but not blocking):**
4. One additional dataset (CNN/DailyMail is standard for summarization)
5. Error analysis with k distribution

**Advantage of ACL/EMNLP:**
- More tolerance for practical contributions without deep theory
- Community familiar with summarization benchmarks and metrics
- Strong engineering and reproducibility are highly valued

**Expected outcome: Likely accept at ACL/EMNLP with the essential fixes above.**

---

## Specific Actionable Recommendations

### 1. EGF Validation (Priority: CRITICAL)

**What to do:**
- Design annotation protocol: "Rate faithfulness on 1-5 scale: Does the summary accurately reflect the source without hallucinations?"
- Recruit 3-5 annotators (graduate students or domain experts)
- Sample 100-150 documents from ArXiv/GovReport dev sets
- Generate summaries with RALFS and 2-3 baselines
- Annotators rate each summary independently
- Compute: Fleiss' κ, mean faithfulness score per system, Spearman/Pearson correlation between human ratings and EGF/BERTScore/ROUGE

**Where to add in paper:**
- New subsection in Results: "Human Evaluation of Faithfulness"
- Table showing: Method, Human Score (mean ± std), ρ with EGF, ρ with BERTScore, ρ with ROUGE-L
- Appendix: Full annotation protocol and examples

**Time estimate:** 2-4 weeks (annotation takes time, but this is non-negotiable for EGF to be a contribution)

---

### 2. Strong Adaptive-k Baselines (Priority: HIGH)

**Learned k-predictor (recommended):**
```python
# Example implementation
from sklearn.linear_model import LogisticRegression

# Extract features per query
features = []
for query in dev_queries:
    scores = rerank(query)
    features.append([
        scores.mean(),
        scores.std(),
        scores[0] - scores[9],  # top-10 gap
        len(query_doc)
    ])

# Labels: best k on dev set per query (oracle)
labels = [best_k_for_query(q, dev_refs) for q in dev_queries]

# Train classifier
clf = LogisticRegression().fit(features, labels)

# Test: predict k, compare to Q(k) formula
```

**Where to add in paper:**
- New row in results tables: "Learned-k FiD"
- Subsection in Results: "Comparison to Alternative k Selection"
- Show: Learned-k vs. Q(k) vs. Oracle-k

**Time estimate:** 1 week (implementation + experiments)

---

### 3. Hybrid Retrieval Ablation (Priority: HIGH)

**What to do:**
- Run experiments with all combinations: D (dense), S (sparse), C (ColBERT), D+S, D+C, S+C, D+S+C
- For each: report ROUGE, mean k, tokens
- Ideally, show both with adaptive-k and fixed-k to isolate interaction

**Where to add:**
- New table in Results or Appendix
- Brief paragraph: "Hybrid fusion improves ROUGE by X points; adaptive k provides additional Y points"

**Time estimate:** 3-5 days (assuming retrieval indexes already exist)

---

### 4. Reframe Claims (Priority: MEDIUM)

**Language to soften:**
- "Principled framework" → "Practical framework" or "Interpretable heuristic"
- "Optimization problem" → "Selection policy based on score distributions"
- "Formal theory" → "Heuristic motivated by information gain"

**Language to keep:**
- "Query-adaptive" ✓
- "Interpretable" ✓
- "Model-agnostic" ✓

**Where to change:**
- Abstract, Introduction (lines 52, 61), Conclusion (line 611)

**Time estimate:** 30 minutes (find-and-replace + light editing)

---

### 5. Add Error Analysis and Computational Cost (Priority: MEDIUM)

**Error analysis:**
```python
# Analyze k selection distribution
k_selected = [select_k(query) for query in test_queries]
plt.hist(k_selected, bins=range(k_min, k_max+1))

# Identify failure cases
for query in test_queries:
    if rouge(ralfs, ref) < rouge(fixed_k_fid, ref):
        print(f"Failure: k={k_selected[query]}, doc_len={len(query)}")
```

**Computational cost:**
```python
import time

# Measure latency
start = time.time()
summary = model.generate(passages[:k])
latency = time.time() - start

# Compare: adaptive-k vs. fixed-k (matched avg)
```

**Where to add:**
- New paragraph in Results: "Efficiency Analysis"
- Appendix: Full k distribution plots

**Time estimate:** 2-3 days

---

### 6. Expand to Additional Dataset (Priority: LOW for ACL, MEDIUM for NeurIPS)

**Easiest option:** CNN/DailyMail (standard summarization benchmark, shorter documents)

**What to do:**
- Chunk documents as in ArXiv/GovReport
- Run full pipeline (retrieval, adaptive-k, generation)
- Report ROUGE, tokens, k distribution

**Where to add:**
- New subsection in Results: "Generalization to News Summarization"
- OR: Appendix if space is tight

**Time estimate:** 1 week (data prep + experiments)

---

## Summary of Recommendation

**Current state:** Borderline reject (5.5/10) due to missing EGF validation, insufficient baselines, and limited evaluation.

**Path forward:**

**For NeurIPS resubmission (challenging):**
- Implement ALL 6 recommendations above
- Estimated time: 6-8 weeks
- Expected score if done well: 6.5-7/10 (Weak Accept)

**For ACL/EMNLP submission (recommended):**
- Implement items 1-3 (CRITICAL and HIGH priority)
- Optionally: items 4-5
- Estimated time: 3-4 weeks
- Expected outcome: Likely Accept

**My strong recommendation:** Target ACL or EMNLP. The paper's practical focus, reproducibility, and clear writing are well-suited to these venues. With EGF validation and stronger baselines, it would be a solid accept. NeurIPS would require substantially more work for a marginal acceptance probability.

---

## Final Comments

This is a well-executed applied paper that solves a real problem. The authors should be proud of the reproducibility effort and clear presentation. The main issue is **overreach in claims vs. evidence**: EGF needs validation, adaptive-k needs stronger baselines, and "principled framework" language needs tempering.

With focused revisions (especially EGF validation), this would be a strong contribution to the summarization community at ACL/EMNLP. The practical impact is real: adaptive evidence selection is useful for production RAG systems.

**Decision: Borderline Reject with encouragement to revise and resubmit to ACL/EMNLP.**

---

**Meta-Reviewer Confidence:** 5/5 (Very High)  
**Recommendation:** Revise and resubmit to ACL 2026 (deadline: Feb 15) or EMNLP 2026 (deadline: May 15)

---

**End of Meta-Review**
