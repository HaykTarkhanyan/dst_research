# Uncertainty-Driven Rule Induction for Dempster-Shafer Classifiers: Paper Plan

**Date:** April 1, 2026

---

## The Core Insight

DSGD++ tells you exactly WHERE it's struggling — rules with high uncertainty are the weak points. You can use this signal to mine *targeted* conjunctive rules for those ambiguous regions. This creates a closed loop: model uncertainty drives rule creation, which reduces uncertainty.

---

## Three Ideas (Combined Into One Paper)

### Idea 1: Uncertainty-Guided Iterative Rule Mining (Main Contribution)

```
UncertaintyGuidedDSGD:

1. Train DSGD++ with standard single-feature rules
2. Identify HIGH-UNCERTAINTY rules (harmonic mean score < threshold)
3. For each weak rule:
   - Find the data points it covers
   - Run RIPPER/SkopeRules on ONLY those points
   - This produces conjunctive rules that resolve the ambiguity
4. Add the new mined rules to the rule set
5. Re-initialize MAFs (using DSGD++ confidence) and re-train
6. Repeat until no high-uncertainty rules remain (or max iterations)
```

**Why this is novel:** Nobody uses DST uncertainty as a *feedback signal* for rule generation. It's a closed loop: model uncertainty drives rule creation, which reduces uncertainty.

**Concrete example on Breast Cancer:**

- Step 1: DSGD++ trains. Rule "clump_thickness > 5" has uncertainty 0.7 — it covers both benign and malignant points.
- Step 2: Look at the 150 points this rule covers. Run RIPPER on just those 150 points.
- Step 3: RIPPER discovers "clump_thickness > 5 AND bare_nuclei > 3 -> malignant" and "clump_thickness > 5 AND bare_nuclei <= 3 -> benign"
- Step 4: These two new rules replace the ambiguous one. Their MAFs get high confidence because they're purer.
- Result: Better separation, lower uncertainty, same or better accuracy.

### Idea 2: Rule Source Ensemble via DST

Instead of choosing ONE rule mining algorithm, treat each algorithm as a **separate evidence source**:

```
Rule Source Ensemble:

1. Generate rules from multiple sources:
   - Source A: Single-feature thresholds (current DSGD)
   - Source B: RIPPER rules
   - Source C: SkopeRules
   - Source D: Decision tree leaf paths (from a shallow tree)

2. Each source produces its own rules + MAFs
3. At prediction time, for a new data point:
   - Collect satisfied rules from ALL sources
   - Combine their MAFs using Dempster's rule

   This is natural -- DST was designed for combining
   evidence from heterogeneous sources!
```

**Why this is interesting:** Each rule mining algorithm has different inductive biases. RIPPER produces ordered rules (early rules are stronger). SkopeRules produces rules with precision/recall guarantees. Decision trees capture different boundaries. Combining them via DST is theoretically justified and practically powerful.

### Idea 3: Confidence-Aware Rule Pruning

Rule mining algorithms tend to produce many rules. Too many rules = slow combination + reduced interpretability. But DSGD++ gives you a principled pruning criterion:

```
After training with mined rules:
- Compute confidence for each rule
- Compute harmonic-mean importance score
- Prune rules below a threshold
- Re-train with remaining rules
```

**The paper angle:** Study the accuracy-interpretability Pareto frontier. Plot accuracy (y-axis) vs. number of rules (x-axis) as you prune more aggressively. Show that confidence-based pruning achieves a better tradeoff than random pruning or frequency-based pruning.

---

## Paper Structure

### Title (working)
*"Uncertainty-Driven Rule Induction for Dempster-Shafer Classifiers"*

### Core Contributions

1. **Iterative uncertainty-guided rule mining** (Idea 1) — the main algorithm
2. **Multi-source rule ensemble via DST** (Idea 2) — shows DST naturally handles heterogeneous rule sources
3. **Confidence-based rule pruning** (Idea 3) — controls interpretability-accuracy tradeoff
4. **DSGD++ confidence initialization generalizes to conjunctive rules** — show that representativeness/confidence still works when rules are more complex

### The Key Figure (Hero Figure)

The paper's main diagram showing the iterative loop:

```
+---------------------------------------------+
|                                             |
|   Data --> Simple Rules --> DSGD++ Train    |
|                                    |        |
|                              Identify        |
|                           High-Uncertainty   |
|                              Rules          |
|                                |             |
|                     +----------+----------+  |
|                     |   Mine targeted     |  |
|                     |  conjunctive rules  |  |
|                     |  for ambiguous      |  |
|                     |  data regions       |  |
|                     +----------+----------+  |
|                                |             |
|                     Add new rules to set     |
|                     Re-initialize MAFs       |
|                          (DSGD++)            |
|                                |             |
|                     <---- Repeat until ------+
|                         uncertainty converges
+---------------------------------------------+
```

---

## Experiments

| Experiment | What it shows |
|-----------|--------------|
| **E1: Rule source comparison** | Single-feature vs. RIPPER vs. SkopeRules vs. C5.0 in DSGD. Table of accuracy, F1, training time, uncertainty, rule count. |
| **E2: Iterative refinement** | Plot uncertainty reduction over iterations of the uncertainty-guided loop. Show it converges in 2-3 iterations. |
| **E3: Rule ensemble** | Single-source DSGD vs. multi-source ensemble. Show ensemble is better, especially on complex datasets. |
| **E4: Pruning Pareto frontier** | Accuracy vs. rule count curve. Compare confidence-based pruning vs. random vs. coverage-based. |
| **E5: Comparison with interpretable baselines** | DSGD + mined rules vs. standalone RIPPER, vs. RuleFit, vs. Bayesian Rule Lists, vs. SIRUS. Use `imodels` Python package for all baselines. |
| **E6: Scaling** | How does it work as datasets get larger? Test on Adult (48K rows), MAGIC Gamma (19K), Bank Marketing (45K). |

### Datasets

Current 5 datasets from DSGD++ PLUS larger ones to show scalability:

| Dataset | Rows | Features | Why include |
|---------|------|----------|-------------|
| Breast Cancer | 699 | 9 | Continuity with DSGD++ paper |
| Brain Tumor | 3762 | 14 | Continuity with DSGD++ paper |
| Heart Disease | 920 | 13 | Classic medical dataset |
| PIMA Diabetes | 768 | 8 | Medical + complex boundaries |
| Adult Income | 48842 | 14 | Scale test |
| MAGIC Gamma | 19020 | 10 | Non-trivial decision boundaries |
| Gaussian/Rectangle/Uniform | 500-1263 | 3 | Controlled scenarios |

### Baselines for E5

| Method | Library | Type |
|--------|---------|------|
| RIPPER (standalone) | `wittgenstein` | Rule induction |
| RuleFit | `imodels` | Rule ensemble |
| Bayesian Rule Lists | `imodels` | Bayesian rules |
| SIRUS | `imodels` | Stable rules from RF |
| SkopeRules (standalone) | `skope-rules` | Bagging + rules |
| Decision Tree | `scikit-learn` | Tree baseline |
| DSGD++ (original) | Your code | Single-feature rules |
| DSGD++ + RIPPER | New | Idea 1 |
| DSGD++ + Ensemble | New | Idea 2 |
| DSGD++ + Pruning | New | Idea 3 |

---

## Implementation

All doable in Python with existing libraries:

```python
# Rule mining sources
from wittgenstein import RIPPER          # RIPPER rules
from skrules import SkopeRules           # SkopeRules
from sklearn.tree import DecisionTreeClassifier  # Tree-path rules
from imodels import RuleFitClassifier    # RuleFit (optional baseline)

# Your existing code
from dsgd import DSGD                    # Your DSGD implementation
```

The main new code to write:
1. **Adapter** to convert mined rules into DSGD rule format (~100 lines)
2. **Iterative uncertainty-guided loop** (~50 lines)
3. **Ensemble combiner** (~30 lines — concatenate rules from different sources)
4. **Pruning logic** (~20 lines)

---

## Assessment

| Aspect | Rating |
|--------|--------|
| **Novelty** | High — uncertainty-driven rule mining loop is new |
| **Feasibility** | High — Python libraries exist, builds directly on DSGD++ |
| **Implementation time** | ~2 weeks |
| **Risk** | Low — even if iterative refinement doesn't help much, the comparison study (E1) and ensemble (E3) are still publishable contributions |
| **Best venues** | Expert Systems with Applications, Information Sciences, ECML-PKDD, Information Fusion |
| **Dependency on multi-class** | None — works with binary classification as-is |

---

## Suggested Order of Work

```
Week 1:
  - Write rule format adapter (RIPPER/SkopeRules -> DSGD rule format)
  - Run E1 (rule source comparison) on all datasets
  - This alone tells you whether mined rules help

Week 2:
  - Implement the iterative uncertainty-guided loop
  - Run E2 (iterative refinement convergence plot)
  - Implement multi-source ensemble
  - Run E3 (ensemble comparison)

Week 3:
  - Implement confidence-based pruning
  - Run E4 (Pareto frontier)
  - Run E5 (comparison with imodels baselines)
  - Run E6 (scaling)

Week 4:
  - Write paper
  - Generate figures (loss curves, Pareto frontier, uncertainty per iteration)
```

---

## Key References

1. Cohen, W.W. (1995). "Fast Effective Rule Induction." *ICML.* — Original RIPPER paper.
2. Quinlan, J.R. (1996). "Improved Use of Continuous Attributes in C4.5." *JAIR.* — C4.5/C5.0 rules.
3. Goix, N. (2020). "SkopeRules." *scikit-learn-contrib.* — SkopeRules.
4. Singh, C. et al. (2021). "imodels: Interpretable Models Python Package." *JOSS.* — Unified baselines.
5. Fumanal-Idocin, J. et al. (2025). "Compact Rule-Based Classifier via Gradient Descent." *arXiv:2502.01375.* — Gradient-based rule optimization (closest related work).
6. Wang, Y. et al. (2025). "Distilling Neural Knowledge into Interpretable Belief Rule Bases." *AAAI.* — Neural-to-BRB distillation.
7. Yang, J.-B. et al. (2006). "Belief Rule-Base Inference Methodology (RIMER)." *IEEE Trans. SMC-A.* — Foundational BRB paper.
8. Zhang, A. et al. (2020). "Rule Reduction for Extended BRB Based on DBSCAN." *IJAR.* — DBSCAN for rule pruning in BRB.
9. Letham, B. et al. (2015). "Interpretable Classifiers Using Rules and Bayesian Analysis." *Annals of Applied Statistics.* — Bayesian Rule Lists.
10. Benard, C. et al. (2021). "SIRUS: Stable and Interpretable Rule Sets." *AISTATS.* — Stable rule extraction from RF.
11. Penafiel, S. et al. (2020). "Applying DST for Developing a Flexible, Accurate and Interpretable Classifier." *ESWA.* — Original DSGD.
12. Tarkhanyan, A. & Harutyunyan, A. (2025). "DSGD++." *J.UCS.* — MAF initialization.
