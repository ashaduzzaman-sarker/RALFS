# NeurIPS Review: RALFS

**Paper ID:** [Redacted]  
**Title:** RALFS: Retrieval-Augmented Long-Form Summarization with Adaptive Passage Selection and Entity Grid Faithfulness

---

## Summary

This paper introduces RALFS, a retrieval-augmented summarization system with two main contributions: (1) adaptive k-selection for Fusion-in-Decoder (FiD) that dynamically determines the number of retrieved passages based on retrieval confidence scores, and (2) Entity Grid Faithfulness (EGF), a reference-aware metric combining entity overlap, transition similarity, and role consistency. The authors evaluate on ArXiv and GovReport, reporting +12% ROUGE-2 improvements and 60% token reduction compared to fixed-k FiD.

---

## Strengths

1. **Clear problem formulation.** The paper identifies a real gap: fixed passage counts in FiD are arbitrary and wasteful. The motivation is well-articulated with the tension between coverage and noise.

2. **Simple, interpretable method.** The adaptive k selector is a post-hoc algorithm based on score dropoffs, making it model-agnostic and practically deployable without retraining generators.

3. **Comprehensive experimental setup.** The authors provide detailed reproducibility information (Table 1), multiple seeds, bootstrap confidence intervals, and versioned configs. This exceeds typical NeurIPS standards.

4. **EGF is grounded in linguistic theory.** The entity grid framework is well-established for coherence; extending it to faithfulness evaluation is a reasonable step.

5. **Writing quality.** The paper is generally well-written with clear structure (Tension→Gap→Insight→Solution→Evidence in the introduction).

---

## Weaknesses

### Major Issues

1. **Limited novelty of adaptive k.** The core idea—"use retrieval scores to select k"—is straightforward engineering. The Q(k) function (cumulative score minus dropoff penalty) is heuristic, not derived from theory. The paper claims this is "principled," but there's no formal justification for why this particular formula optimizes information gain. The comparison to cascade retrieval (Section 2.2.1) helps differentiate scope but doesn't address the fundamental incrementality: you're applying a threshold policy to already-ranked passages.

   **Critical question:** Why is this a research contribution rather than a hyperparameter scheduling trick? What prevents me from simply grid-searching k per dataset and getting similar results?

2. **Weak baselines for adaptive k.** The paper compares against *fixed* k ∈ {5,10,15,20} but not against:
   - Learned k-predictors (e.g., train a small MLP on retrieval features to predict k)
   - Confidence-based thresholds (stop when score < threshold, standard in IR)
   - Oracle k (best k per query in hindsight)
   
   Without these, it's unclear whether the +12% gain comes from the *specific* Q(k) formula or simply from "not using a fixed k." The ablation in scripts/run_ablation_study.py suggests testing different strategies (score_dropoff, confidence), but results aren't reported.

3. **EGF validation is insufficient.** The paper claims Spearman ρ = 0.612 vs. BERTScore ρ = 0.447 on human judgments (Section 2.4.2, line 134) but provides no table, no significance test, no sample size, no inter-annotator agreement. The reference to "Table~\ref{tab:human_eval}" is undefined. How many annotators? What was the annotation protocol? Were they blinded? This is a core claim and needs rigorous validation.

4. **No error analysis or failure cases.** The Discussion mentions failure modes abstractly (entity-sparse texts, flat scores) but provides no quantitative breakdown. How often does the selector over-/under-select? What fraction of queries get k_min vs. k_max? Are there systematic biases by document length, genre, or query type?

5. **Compute savings not rigorously measured.** The paper claims 60% token reduction (140 vs. 350 tokens on ArXiv) but doesn't report:
   - Wall-clock latency reduction (encoding + decoding)
   - Memory usage reduction
   - FLOPs or GPU-hours saved
   
   Token count is a proxy, but practical impact requires end-to-end measurements. The "60% reduction" could be offset by overhead from the selector or batch padding inefficiencies.

6. **Hybrid retrieval is orthogonal and under-ablated.** The paper uses dense+sparse+ColBERT fusion with RRF, but the contribution of each component to the final ROUGE gain is unclear. Section 2.5 claims hybrid outperforms single methods by 3-5 ROUGE-L points, but no table shows dense-only, sparse-only, ColBERT-only, or pairwise fusion results. Since adaptive k depends on score distributions, separating the gain from fusion vs. adaptive k is critical.

7. **Limited datasets and domains.** Only ArXiv (scientific) and GovReport (government) are tested—both are formal, structured documents. What about news (CNN/DailyMail), Wikipedia, dialogue, or multilingual settings? The generalization claim in Section 4.2 is speculative without evidence.

### Minor Issues

8. **Inconsistent notation.** The paper uses k, k*, k^*, and k_{\max} inconsistently. In equation (7), k^* is the selected value, but in the text, both k and k^* appear.

9. **Missing related work.** The paper doesn't cite:
   - Dynamic evaluation (Krause et al., 2018) for adaptive compute
   - Threshold-based stopping in neural IR (e.g., confidence calibration in dense retrieval)
   - Recent RAG work (RETA, Self-RAG, IRCoT) that also address evidence selection

10. **Hyperparameter sensitivity not shown.** How sensitive is performance to λ, k_min, k_max? Table 1 lists these but provides no ablation. If λ = 0, does the method degrade to fixed k? If λ is large, does it always select k_min?

11. **Statistical testing details missing.** The paper mentions paired bootstrap tests (1k replicates) but doesn't report p-values, effect sizes (Cohen's d), or win/tie/loss counts. The claim "consistent gains" (line 555) needs quantitative support.

12. **EGF weights (α, β, γ) are arbitrary.** Section 3.5.4 sets α = 0.4, β = 0.4, γ = 0.2 "empirically" but provides no ablation or justification. Were these tuned on a held-out set? Grid-searched? Uniform weighting (α = β = γ = 1/3) might work equally well.

---

## Questions for Authors

1. **Adaptive k vs. learned k:** Can you compare against a simple learned predictor (e.g., logistic regression or small MLP) that predicts k from retrieval features (mean score, variance, top-k gap)? This would clarify whether the hand-crafted Q(k) formula is necessary.

2. **EGF human evaluation:** Please provide full details: annotator count, agreement (Fleiss' κ), annotation protocol, sample size, significance tests, and the actual Table~\ref{tab:human_eval}.

3. **Oracle k analysis:** What is the performance if you select the *best* k per query in hindsight (using dev ROUGE)? This upper bound would quantify how much room for improvement remains.

4. **Hybrid retrieval ablation:** Can you provide a table with all single-method and pairwise fusion results? E.g., Dense-only, Sparse-only, ColBERT-only, Dense+Sparse, Dense+ColBERT, Sparse+ColBERT, All-three?

5. **Failure case analysis:** What fraction of queries select k_min vs. k_max? Are there systematic failures (e.g., over-selection on entity-sparse documents)?

6. **Computational cost:** Can you report wall-clock latency, GPU memory, and FLOPs for adaptive k vs. fixed k (with matched average k)?

7. **Hyperparameter sensitivity:** Can you show performance curves for λ ∈ [0, 2], k_min ∈ [1, 10], k_max ∈ [10, 50]?

8. **Generalization:** Have you tested on other datasets (CNN/DailyMail, XSum, multi-document summarization)?

---

## Detailed Comments

### Introduction
- Line 42: "forcing systems to retrieve" is too strong—retrieval is a design choice, not forced by context windows.
- Line 52: "Formalizing this as an optimization problem" is misleading—Q(k) is a heuristic, not derived from optimization.
- Line 75: The claim "ρ > 0.6" needs a reference to the missing table.

### Related Work
- Section 2.2.1 (Cascade vs. Adaptive k) is helpful but overstates the difference. Both apply post-hoc policies to ranked lists; the scope differs (ranking vs. decoding), but the mechanism is similar.
- Section 2.4.2: The claim "domain transfer" (entity grids → faithfulness) is reasonable but under-validated.

### Method
- Equation (6): Why is the penalty max(0, s_{k-1} - s_k) rather than a smoothed gap or variance? This seems brittle to noisy scores.
- Line 240: "empirically set λ = 1.0"—on what data? Dev set? If so, how sensitive is this?
- Algorithm 1 (EGF): The pseudocode is clear, but complexity analysis is missing. What is the runtime of entity extraction and grid construction? For long documents (10K tokens), is this practical?

### Experimental Setup
- Table 1 is excellent and exceeds typical standards. Minor: "where to document" column could specify exact file paths.
- Section 4.4: "Early stopping monitors validation ROUGE-L"—does this introduce selection bias? If you stop early based on ROUGE, reporting ROUGE as the primary metric is circular.

### Results
- Section 5.1: The tables show RALFS is best, but without significance markers (†/‡) or p-values, the magnitude of improvement is unclear. Is +0.018 ROUGE-2 (0.242 vs. 0.224) reliably different?
- Section 5.2: "Why RALFS Improves Quality" is speculative. The claim "evidence noise" is reasonable but not directly measured. Can you show attention entropy or KL divergence between RALFS and fixed-k decoder states?
- Section 5.3: The efficiency-quality trade-off narrative is good, but without latency/memory data, it's incomplete.

### Discussion
- Section 6.1: The RAG implications are oversold. "First-class control" suggests a principled framework, but the method is a post-hoc heuristic.
- Section 6.3: The trade-offs are honest, which is commendable. However, these should be backed by empirical evidence (e.g., calibration curves, k variance histograms).

### Conclusion
- The claim "principled framework" (line 611) is too strong given the heuristic nature of Q(k).

---

## Scores

**Novelty / Originality:** 4 / 10  
*The adaptive k idea is straightforward; Q(k) is a simple heuristic. EGF extends entity grids predictably. Incremental over existing RAG and coherence work.*

**Technical Quality / Soundness:** 5 / 10  
*Experimental setup is rigorous (reproducibility, seeds, CIs), but key ablations are missing (learned k, oracle k, hybrid retrieval breakdown). EGF validation is inadequate. No error analysis or computational cost measurements.*

**Clarity / Presentation:** 7 / 10  
*Writing is generally clear and well-structured. However, notation inconsistencies, missing table references, and unsupported claims (e.g., ρ = 0.612) detract. The Discussion is honest about limitations.*

**Significance / Impact:** 5 / 10  
*The practical utility of adaptive k is real but limited to a narrow context (FiD + long-document summarization). EGF is unlikely to replace existing metrics without stronger validation. Limited datasets and domains restrict generalization claims.*

**Reproducibility:** 8 / 10  
*Excellent: Table 1, versioned configs, seeds, CIs. However, missing details on EGF annotation and some ablation scripts mentioned but not run.*

**Overall Score:** 5 / 10  
**Confidence:** 4 / 5 (High)

---

## Recommendation

**Reject**

**Justification:**  
While the paper addresses a real problem (arbitrary k in FiD) and provides strong reproducibility artifacts, the contributions are **incremental** and **under-validated**. The adaptive k selector is a straightforward heuristic without theoretical grounding or strong baselines (no learned k, no oracle k). The claimed +12% ROUGE-2 gain is impressive, but it's unclear whether this comes from adaptive selection or simply from not using a bad fixed k—proper ablations are missing.

The EGF metric is interesting but critically under-validated: the human evaluation claim (ρ = 0.612) lacks supporting evidence (no table, no protocol, no agreement scores). Without this, EGF remains speculative.

The paper is well-written and the reproducibility effort is commendable, but **novelty is insufficient for NeurIPS**, and key experiments are missing. I recommend:

1. **Reject and resubmit** with:
   - Strong baselines: learned k-predictor, confidence thresholds, oracle k
   - Full EGF human evaluation with protocol, agreement, significance tests
   - Comprehensive hybrid retrieval ablation (all single/pairwise combinations)
   - Error analysis: k selection distribution, failure cases, systematic biases
   - Computational cost: latency, memory, FLOPs
   - Broader evaluation: 3+ additional datasets (news, dialogue, multilingual)

2. **Alternative venue:** Consider *ACL or EMNLP* (better fit for applied NLP systems with strong engineering) or *TMLR* (thorough but incremental contributions).

---

## Ethical Concerns

None identified. The paper does not raise issues related to bias, fairness, privacy, or dual use. The approach is domain-agnostic and does not target sensitive populations.

---

## Reproducibility Checklist (for reference)

- [x] Code, data, and models will be released (implied by configs)
- [x] Hyperparameters disclosed (Table 1)
- [x] Seeds and statistical testing (mentioned in Section 4.4)
- [x] Compute resources specified (A100 GPUs)
- [ ] Human evaluation protocol (missing details)
- [ ] Full ablation studies (partial)

---

## Reviewer Expertise

Machine Learning (RAG, retrieval-augmented generation), NLP (summarization, evaluation metrics), Neural Information Retrieval. Highly familiar with FiD, entity grids, and adaptive computation.

---

**End of Review**
