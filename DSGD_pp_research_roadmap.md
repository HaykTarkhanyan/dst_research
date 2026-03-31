# DSGD++ Research Roadmap: Comprehensive Literature Review & Future Directions

**For:** Aik Tarkhanyan & Ashot Harutyunyan
**Date:** April 1, 2026
**Based on:** Systematic review of 90+ papers across arXiv, Semantic Scholar, IEEE, Springer, Elsevier, and Google Scholar

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Literature Review by Theme](#2-literature-review-by-theme)
   - 2.1 [DSGD Lineage & DST-Based Classifiers](#21-dsgd-lineage--dst-based-classifiers)
   - 2.2 [Mass Function Initialization & Learning](#22-mass-function-initialization--learning)
   - 2.3 [Evidential Deep Learning (EDL)](#23-evidential-deep-learning-edl)
   - 2.4 [Uncertainty Measures in DST](#24-uncertainty-measures-in-dst)
   - 2.5 [Rule Mining + Evidence Theory](#25-rule-mining--evidence-theory)
   - 2.6 [Multi-Class/Multi-Label Scalability](#26-multi-classmulti-label-scalability)
   - 2.7 [Conformal Prediction + DST](#27-conformal-prediction--dst)
   - 2.8 [Active Learning with Belief Functions](#28-active-learning-with-belief-functions)
   - 2.9 [Federated/Distributed DST](#29-federateddistributed-dst)
   - 2.10 [Alternatives to Dempster's Combination Rule](#210-alternatives-to-dempsters-combination-rule)
3. [Strategic Gaps Identified](#3-strategic-gaps-identified)
4. [Research Directions (Ranked)](#4-research-directions-ranked)
5. [Full Reference List](#5-full-reference-list)

---

## 1. Executive Summary

DSGD++ occupies a unique position in the ML landscape: it is the **only DST-based classifier that uses gradient descent to optimize mass assignment functions (MAFs) while providing inherent interpretability through human-readable rules**. The clustering-based MAF initialization is a genuinely novel contribution — the literature confirms that MAF initialization is severely understudied.

Three strategic observations from the review:

1. **MAF initialization is a real gap.** Only three approaches exist: random (Peñafiel 2020), statistical distribution (Sedláček 2024), and clustering-based (DSGD++ 2025). There is significant room for theoretical and empirical contributions.

2. **The harmonic-mean uncertainty measure has no direct precedent.** While Deng entropy and its extensions are well-studied, no prior work proposes a rule-importance measure combining uncertainty and mass ratio via harmonic mean.

3. **DSGD++ bridges two worlds.** Evidential Deep Learning (Sensoy 2018+) uses DST in neural networks but sacrifices interpretability. Classical DST classifiers (EK-NN, Denoeux 2000) lack optimization. DSGD++ combines gradient-based learning with interpretable rules — a unique middle ground.

---

## 2. Literature Review by Theme

### 2.1 DSGD Lineage & DST-Based Classifiers

| # | Paper | Year | Venue | Citations | Relevance |
|---|-------|------|-------|-----------|-----------|
| P1 | Peñafiel et al., "Applying DST for Developing a Flexible, Accurate and Interpretable Classifier" | 2020 | Expert Systems with Applications | ~90 | **Foundational DSGD paper.** Introduces rule-based DST classifier with MAF optimization via gradient descent. Random MAF init (0.8 to full set, 0.2 random to singletons). |
| P2 | Peñafiel, "Interpretable Method for General Classification Using DST" (Thesis) | 2021 | Universidad de Chile | — | Generalizes DSGD to multi-class; applies to stroke risk prediction. |
| P3 | Baloian et al., "Embedded Interpretable Regression using DST" | 2024 | CODASSCA | — | Extends DSGD to regression (EVREG) using evidential regression with discount functions. |
| P4 | Baloyan et al., "Empirical Analysis of Feature Engineering for DS Classifier" | 2024 | CODASSCA | — | Studies robustness to class imbalance and feature interactions in DSGD. |
| P5 | Valdivia et al., "Interpretable Clustering Using DST" | 2024 | CODASSCA / J.UCS 2025 | — | Extends DSGD to clustering (DSClustering). Silhouette=0.73, best among tested methods. |
| P6 | Tarkhanyan & Harutyunyan, "DSGD++" | 2025 | J.UCS | 0 | **Your paper.** Clustering-based MAF init, 1.6x speedup, 2.1x uncertainty reduction. |
| P7 | Tarkhanyan & Harutyunyan, "Improving DSGD with MAF Initialization" | 2024 | CODASSCA | — | Preliminary version using arithmetic mean instead of KMeans. |
| P8 | Sedláček & Bartoš, "Fusing Heterogeneous Data for Network Asset Classification" | 2024 | IEEE NOMS | — | Statistical distribution-based MAF init for network classification. |
| P9 | Barseghyan, "Explainable AI for breast cancer risk prediction" | 2025 | ML for Computational Science | 1 | Evaluates accuracy-explainability tradeoff; references DSGD approach. |

**Key insight:** The DSGD "family" now covers classification (P1), regression (P3), clustering (P5), and feature engineering (P4). DSGD++ (P6) is the only one addressing initialization.

### 2.2 Mass Function Initialization & Learning

| # | Paper | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| P10 | Denoeux, "A Neural Network Classifier Based on DST" | 2000 | IEEE Trans. SMC-A | Evidential neural network: each neighbor provides a BPA based on distance; init is distance-based, not learned. |
| P11 | Denoeux, "Logistic Regression, Neural Networks and DST: A New Perspective" | 2019 | Knowledge-Based Systems | Shows softmax layers can be reinterpreted as Dempster's combination rule. Mass functions emerge from network weights — learned end-to-end, not initialized. |
| P12 | Erber & Laor, "Laor Initialization: Weight Init via KMeans Clustering" | 2025 | Big Data and Cognitive Computing | **Direct analogue to DSGD++.** Uses KMeans clustering on random initial weights to find optimal NN initialization. 24% training time reduction. Same insight: clustering-based init → faster convergence. |
| P13 | Ash & Adams, "On Warm-Starting Neural Network Training" | 2020 | NeurIPS | Shows naive warm-starting can hurt generalization; proposes shrink+perturb fix. Relevant caveat for MAF warm-starting. |
| P14 | Zhang et al., "New Rule Reduction for Extended BRB Based on DBSCAN" | 2020 | IJAR | Uses DBSCAN to reduce rules in belief rule bases — closest existing work to using DBSCAN in a DST classifier context. |
| P15 | Liu et al., "DS Fusion-Based Approach for Malware Detection" | 2025 | Mathematics (MDPI) | Comprehensive evaluation of mass function construction methods and combination rules for ensemble classification. |

**Gap confirmed:** No paper besides DSGD++ proposes clustering-based initialization of mass functions for DST classifiers. The Laor init paper (P12) does the same for neural networks, but not for DST.

### 2.3 Evidential Deep Learning (EDL)

| # | Paper | Year | Venue | Citations | Relevance |
|---|-------|------|-------|-----------|-----------|
| P16 | Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty" | 2018 | NeurIPS | ~1400 | **Seminal EDL paper.** Places Dirichlet prior over class probabilities via subjective logic/DST. Single forward pass yields epistemic + aleatoric uncertainty. |
| P17 | Amini et al., "Deep Evidential Regression" | 2020 | NeurIPS | ~700 | Extends EDL to regression; learns evidential distributions over continuous targets. |
| P18 | Charpentier et al., "Posterior Network" | 2020 | NeurIPS | ~300 | Normalizing flow on Dirichlet posterior; stronger theoretical guarantees than EDL. |
| P19 | Tong, Xu & Denoeux, "Evidential Classifier Based on DST and Deep Learning" | 2021 | Neurocomputing | ~80 | CNN + DS layer converting features to mass functions + expected utility layer. Multi-class image classification with set-valued outputs. |
| P20 | Gong et al., "Evidential Deep Learning on ImageNet-Scale" | 2024 | — | — | Scales EDL to large-scale vision tasks, addressing prior limitations. |
| P21 | Kandemir et al., "Are EDL Uncertainty Capabilities a Mirage?" | 2024 | NeurIPS | — | **Critical paper.** Questions whether EDL's uncertainty estimates are truly faithful. Important for positioning DSGD++ as an alternative. |
| P22 | Duan et al., "Fisher Information-based EDL" | 2023 | arXiv | — | Improves EDL calibration using Fisher information. |
| P23 | Chen et al., "Comprehensive Survey on EDL" | 2024 | arXiv / IEEE TPAMI | — | **Most comprehensive EDL survey.** 163 references covering the full landscape. |
| P24 | Flexible EDL (F-EDL) | 2025 | OpenReview | — | Extends EDL with flexible Dirichlet distributions for more expressive uncertainty. |

**Connection to DSGD++:** EDL (P16) and DSGD++ both use DST for classification, but EDL sacrifices interpretability for scalability. A hybrid (deep features → DSGD++ rules) would combine strengths of both.

### 2.4 Uncertainty Measures in DST

| # | Paper | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| P25 | Deng, "Deng Entropy" | 2016 | Chaos, Solitons & Fractals | Generalization of Shannon entropy for BPAs. The standard alternative to "mass of complete set." |
| P26 | Chen & Deng, "Complex Deng Entropy" | 2024 | Engineering Applications of AI | Extends Deng entropy to complex-valued evidence; applicable to ordered frames. |
| P27 | Extended Fractional Deng Entropy | 2024 | Communications in Statistics | Two new extended fractional Deng entropy measures with applications. |
| P28 | Belief Rényi Entropy for DST | 2023 | Communications in Statistics | New uncertainty measure inspired by Rényi entropy, applicable to BPAs. |
| P29 | Jirousek & Shenoy, numerical comparison of uncertainty measures | 2023 | IJAR | Comprehensive numerical comparison of DST uncertainty measures. |
| P30 | Pan & Deng, "Belief interval-based total uncertainty measure" | 2023 | Information Sciences | Uses belief intervals [Bel, Pl] as bounds; improves on Deng entropy. |
| P31 | Improved Belief Entropy based on Deng + belief interval | 2020 | Entropy (MDPI) | Combines Deng entropy with belief intervals for better uncertainty measurement. |

**Your harmonic-mean measure** (combining 1-uncertainty and mass ratio) has no direct precedent in this literature. Most work extends Deng entropy or uses belief intervals. Your approach is **complementary** — it measures rule quality for classification specifically, while these measure general BPA uncertainty.

### 2.5 Rule Mining + Evidence Theory

| # | Paper | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| P32 | Yang et al., "Belief Rule-Base Inference Methodology (RIMER)" | 2006 | IEEE Trans. SMC-A | **Foundational BRB paper.** IF-THEN rules with belief degrees, combined via evidential reasoning. 1000+ citations. |
| P33 | Zhang et al., "Rule Reduction for Extended BRB Based on DBSCAN" | 2020 | IJAR | Uses DBSCAN to prune unused rules in BRB; parameter learning improves accuracy. |
| P34 | BRB Optimization with Activation Rate | 2023 | Scientific Reports | Novel BRB optimization pruning rules that are never activated during training. |
| P35 | Multi-parameter Optimized BRB for Student Performance | 2026 | Scientific Reports | Recent BRB optimization maintaining interpretability. |
| P36 | Fumanal-Idocin et al., "Compact Rule-Based Classifier via Gradient Descent" (FRR) | 2025 | arXiv | **Highly relevant.** Fuzzy Rule Reasoner with end-to-end differentiable optimization. 5% accuracy over RIPPER, 90% more compact rules. Demonstrates gradient-based rule optimization is a hot topic. |
| P37 | Wang et al., "Distilling Neural Knowledge into Interpretable Belief Rule Bases" | 2025 | AAAI/OpenReview | Neural-to-BRB knowledge distillation. NN teacher → BRB student, all parameters optimized via gradient descent. |
| P38 | "A Class Inference Scheme with DST for Learning Fuzzy-Classifier Systems" | 2025 | arXiv | Uses DST for class inference in fuzzy rule systems. |
| P39 | Cohen, "Fast Effective Rule Induction" (RIPPER) | 1995 | ICML | ~5000 | Foundational rule mining paper. Python impl: `wittgenstein` library. |
| P40 | Goix, "SkopeRules" | 2020 | scikit-learn-contrib | Bagging + decision tree rule extraction with diversity selection. Python package. |
| P41 | Singh et al., "imodels: Interpretable Models Python Package" | 2021 | JOSS | Unified scikit-learn-compatible package with RuleFit, BRL, SkopeRules, FIGS. |

**Opportunity:** Combining mined rules (P39-P41) with DSGD++'s confidence-based MAF initialization is unexplored. The BRB literature (P32-P35) provides theoretical grounding, and FRR (P36) shows gradient-based rule optimization is gaining traction.

### 2.6 Multi-Class/Multi-Label Scalability

| # | Paper | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| P42 | Denoeux, "EK-NN Classification Rule Based on DST" | 1995 | IEEE Trans. SMC | Evidential k-NN; handles multi-class naturally but faces exponential focal element growth. |
| P43 | Younes et al., "Evidence-Theoretic k-NN for Multi-Label Classification" | 2009 | SUM (LNCS) | Extends EK-NN to multi-label; frame of discernment = power set of label set. |
| P44 | Denoeux & Masson, "Evidential Reasoning in Large Partially Ordered Sets" | 2012 | Annals of OR | **Key scalability paper.** DST on lattice structures (not full power set) enables tractable multi-label classification. |
| P45 | Wilson, "Monte-Carlo Algorithm for DST Belief" | 1993 | UAI | Approximate DS computation in O(linear) time when conflict is bounded. |
| P46 | Bauer, "Approximations for Decision Making in DST" | 1997 | UAI | Systematic comparison: Bayesian approx, k-l-x methods, D1 algorithm. |
| P47 | Skau, "Open World DST Using Complementary Sets" | 2023 | AISTATS | Handles open-world (unknown classes at test time) without enumerating full power set. |

**For DSGD++:** Multi-class extension is the most requested feature. Options: (a) one-vs-rest decomposition (simplest), (b) lattice-based frames (P44), (c) focal element pruning via confidence scores.

### 2.7 Conformal Prediction + DST

| # | Paper | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| P48 | Hofer & Krempl, "DST + Conformal Prediction" | 2024 | ECML-PKDD | **Key paper.** Combines DST mass functions with conformal prediction for set-valued predictions with formal coverage guarantees and smaller prediction sets. |
| P49 | Karimi et al., "Evidential Uncertainty Sets in Deep Classifiers Using Conformal Prediction" (ECP) | 2024 | COPA (PMLR) | Uses EDL-derived evidence as nonconformity score in conformal prediction. Shows EDL uncertainty leads to more efficient prediction sets. |
| P50 | Angelopoulos & Bates, "Conformal Prediction Tutorial" | 2021 | arXiv | ~1200 | Standard reference for conformal prediction methodology. |

**High-value opportunity:** Wrapping DSGD++ in conformal prediction (using its uncertainty measure as nonconformity score) would add distribution-free guarantees with minimal implementation effort.

### 2.8 Active Learning with Belief Functions

| # | Paper | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| P51 | Horanyi & Hullermeier, "Evidential Uncertainty Sampling for Active Learning" | 2024 | Machine Learning (Springer) | **Directly applicable.** Proposes strategies using belief function decomposition (epistemic vs. aleatoric) for active learning query selection. |
| P52 | Horanyi & Hullermeier, "DS-Inspired Uncertainty Measures for Active Learning" | 2023 | BELIEF (LNCS) | New uncertainty measures derived from belief functions, specifically designed for active learning. |
| P53 | Nguyen et al., "Uncertainty in Active Learning" | 2021 | Machine Learning | Broader survey on uncertainty-driven active learning strategies. |

**For DSGD++:** The rule-level uncertainty in DSGD++ naturally identifies regions where the model is uncertain → ideal for active learning query selection.

### 2.9 Federated/Distributed DST

| # | Paper | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| P54 | Chen et al., "Fusion Strategies for Federated Decision Making" | 2023 | arXiv | Reviews how to aggregate evidential information from distributed sources. |
| P55 | Li et al., "Multisensor Data Fusion in IoT via DST" | 2023 | Sensors (MDPI) | Improved evidence distance for IoT fusion; addresses conflicting evidence. |
| P56 | DS-based Emergency Management Fusion | 2024 | Information Fusion (Elsevier) | Systematic review of DST for fusing information in distributed systems. |

**Major gap:** Federated DSGD++ is essentially unexplored. Dempster's combination rule is inherently designed for fusing independent evidence — a natural fit for aggregating models from different clients.

### 2.10 Alternatives to Dempster's Combination Rule

| # | Paper | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| P57 | Yager, "On the Dempster-Shafer Framework and New Combination Rules" | 1987 | Information Sciences | ~2000 | Assigns conflicting mass to full set (ignorance) instead of normalizing. |
| P58 | Dubois & Prade, "Representation and Combination of Uncertainty with Belief Functions and Possibility Measures" | 1988 | Computational Intelligence | Disjunctive combination: conflicting mass goes to union of focal elements. |
| P59 | Murphy, "Combining Belief Functions When Evidence Conflicts" | 2000 | Decision Support Systems | Averaging mass functions before combination; reduces conflict sensitivity. |
| P60 | Lefevre, Colot & Vannoorenberghe, "Belief Functions Combination and Conflict Management" | 2002 | Information Fusion | Comprehensive comparison of combination rules. |
| P61 | FNBT: Full Negation Belief Transformation | 2025 | arXiv | Recent open-world belief transformation addressing limitations of Dempster's rule. |

**For DSGD++:** Currently uses Dempster's rule. Testing alternatives (especially Murphy averaging for handling conflicting rules) could improve robustness.

---

## 3. Strategic Gaps Identified

Based on this review, the following are **genuine research gaps** where DSGD++ is well-positioned to contribute:

| Gap | Evidence | Impact |
|-----|----------|--------|
| **MAF initialization theory** | Only 3 methods exist (random, statistical, clustering). No convergence analysis. | High — foundational |
| **Federated DST classification** | No papers combine federated learning with DST classifiers | High — novel direction |
| **DSGD++ + conformal prediction** | Only 2 papers (P48, P49) connect DST with CP; neither uses DSGD | High — adds guarantees |
| **Rule mining → DSGD** | BRB literature is separate from DSGD; no paper combines RIPPER/SkopeRules with DSGD | Medium-High |
| **Axiomatic uncertainty for DST classifiers** | Many entropy measures for general BPAs, none specific to rule-based classification | Medium-High |
| **Active learning with DSGD++** | P51-P52 propose theory; no implementation with DSGD-type models | Medium |
| **Online/streaming DSGD++** | No online DST classifier exists | Medium |
| **Multi-class DSGD++** | DSGD family is binary only (except Peñafiel thesis) | High — prerequisite |

---

## 4. Research Directions (Ranked)

### Tier 1: High Impact, Near-Term (3-6 months)

#### 1. Multi-Class DSGD++ via Hierarchical MAF Initialization
- **Why:** Removes the most cited limitation. Prerequisite for most other directions.
- **How:** One-vs-rest decomposition (each binary sub-problem uses DSGD++ init), or hierarchical class taxonomy with MAFs initialized per level.
- **Novelty:** Medium | **Feasibility:** High
- **Venues:** Expert Systems with Applications, Information Fusion, ECML-PKDD

#### 2. DSGD++ with Conformal Prediction Guarantees
- **Why:** Adds distribution-free, finite-sample coverage guarantees to DSGD++ uncertainty. Only 2 papers (P48, P49) connect DST+CP; neither uses DSGD.
- **How:** Use DSGD++ uncertainty (or your harmonic mean measure) as nonconformity score. Calibrate on held-out set. Key question: does DSGD++ yield *smaller* prediction sets than CP wrappers on black-box models?
- **Novelty:** High | **Feasibility:** High (MAPIE library)
- **Venues:** AISTATS, NeurIPS, JMLR

#### 3. Rule Mining Integration (RIPPER/SkopeRules → DSGD++)
- **Why:** Explicitly promised in your future work. Existing rules are single-feature thresholds; mined rules capture feature interactions.
- **How:** Replace DSGD rule generation with `wittgenstein` (RIPPER) or `skope-rules`. Apply same clustering-based confidence for MAF init. Compare accuracy, rule count, training time, uncertainty.
- **Novelty:** Medium | **Feasibility:** High (Python packages exist)
- **Venues:** Expert Systems with Applications, Information Sciences

#### 4. Comprehensive UQ Benchmark: DSGD++ vs. Modern Methods
- **Why:** DSGD++ has never been compared to MC Dropout, Deep Ensembles, Conformal Prediction, or EDL on a level playing field.
- **How:** OpenML-CC18 benchmark (15-20 tabular datasets). Metrics: accuracy, ECE, Brier score, selective prediction, OOD detection AUROC. Also measure interpretability (rule count vs. parameters).
- **Novelty:** Medium | **Feasibility:** High
- **Venues:** NeurIPS Datasets & Benchmarks, Machine Learning Journal, TMLR

### Tier 2: High Impact, Medium-Term (6-12 months)

#### 5. Convergence Theory for Confidence-Based MAF Initialization
- **Why:** Transforms empirical findings into theoretical contribution. No convergence theory for DSGD optimization exists.
- **How:** Formalize MAF optimization as minimizing loss over probability simplex. Show confidence-based init starts in lower-gradient region. Derive convergence rate bounds vs. random init.
- **Novelty:** High | **Feasibility:** Medium
- **Venues:** JMLR, ICML, COLT

#### 6. Federated DSGD++ for Privacy-Preserving Interpretable Classification
- **Why:** Dempster's rule is *designed* for fusing independent evidence — perfect fit for federated aggregation. Essentially unexplored (Gap confirmed in §2.9).
- **How:** Each client computes local rules + MAFs. Server aggregates via Dempster's rule. Study non-IID robustness and privacy leakage from aggregated MAFs.
- **Novelty:** High | **Feasibility:** Medium
- **Venues:** AAAI, IJCAI, IEEE TIFS

#### 7. Generalized Uncertainty Measures: Beyond Harmonic Mean
- **Why:** Your harmonic mean measure is novel but has limitations (requires min-max normalization; equal weighting). A parametric family would be more flexible.
- **How:** Define $I_\beta = (1+\beta^2) \cdot \frac{U' \cdot R'}{\beta^2 U' + R'}$ (analogue of $F_\beta$ score). Study axioms: monotonicity, subadditivity, permutation invariance. Compare against Deng entropy, belief Rényi entropy, belief interval measures.
- **Novelty:** High | **Feasibility:** Medium
- **Venues:** IJAR, Information Sciences, Artificial Intelligence

#### 8. Active Learning with DSGD++ Uncertainty
- **Why:** DSGD++ rule-level uncertainty naturally identifies ambiguous regions. Horanyi & Hullermeier (P51) provide the theoretical framework; no one has applied it to DSGD-type models.
- **How:** Use DSGD++ mass functions to compute epistemic uncertainty per sample. Select most uncertain samples for labeling. Compare strategies: mass of full set, your harmonic mean, Klir uncertainty (P51).
- **Novelty:** Medium | **Feasibility:** High
- **Venues:** Machine Learning, ECML-PKDD

### Tier 3: Ambitious, Long-Term (12+ months)

#### 9. Non-Singleton Focal Elements via Confidence-Guided Selection
- **Why:** Unlocks full DST expressiveness (e.g., mass to {cancer, pre-cancer} jointly). The $O(2^n)$ problem makes this hard, but confidence scores can guide which subsets matter.
- **How:** If two classes have overlapping representativeness (close centroids), create composite focal element for their union. Keep frame sparse.
- **Novelty:** High | **Feasibility:** Medium
- **Venues:** IJAR, Information Fusion

#### 10. Hybrid Deep Learning + DSGD++ (Neural Feature Extractor → Interpretable DST Head)
- **Why:** Extends DSGD++ beyond tabular data. "Neural encoder + interpretable head" is a growing paradigm (concept bottleneck models).
- **How:** Train ResNet/transformer for feature extraction → discretize embeddings into rules → DSGD++ with confidence init. Compare against concept bottleneck models.
- **Novelty:** Medium | **Feasibility:** Medium
- **Venues:** AAAI, NeurIPS, Pattern Recognition

#### 11. Online/Streaming DSGD++ for Concept Drift
- **Why:** No online DST classifier exists. Streaming applications (fraud detection, network monitoring) need incremental updates.
- **How:** Online KMeans for representativeness updates; running-average confidence; warm-start MAFs from previous values. Detect concept drift via representativeness distribution shift.
- **Novelty:** High | **Feasibility:** Medium
- **Venues:** Machine Learning, KDD, ECML-PKDD

#### 12. Evidential Transfer Learning: MAF Transfer Across Related Tasks
- **Why:** If DSGD++ is trained on one task, can MAFs transfer to a related task?
- **How:** Define task similarity based on feature/rule overlap. For shared rules, transfer learned MAFs. For new rules, interpolate confidence from nearest source rules.
- **Novelty:** High | **Feasibility:** Medium
- **Venues:** ECML-PKDD, AAAI, IEEE TNNLS

### Tier 4: Applications

#### 13. DSGD++ for Medical Diagnosis
- **Why:** Healthcare demands both interpretability and calibrated uncertainty. DSGD++ rules are readable; MAFs quantify per-rule confidence.
- **Datasets:** MIMIC-III tabular, Pima Diabetes, Heart Disease, Dermatology
- **Venues:** J. Biomedical Informatics, AI in Medicine, CHIL

#### 14. DSGD++ for Cybersecurity Intrusion Detection
- **Why:** IDS needs explainability (analysts must understand alerts). Three-way decision (normal/attack/uncertain-escalate) using DST uncertainty.
- **Datasets:** NSL-KDD, CICIDS2017, UNSW-NB15
- **Venues:** Computers & Security, IEEE TDSC

#### 15. Sensitivity Analysis of Representativeness Hyperparameters
- **Why:** zScoreThreshold=2, minPoints=2*features, maxEps=20, step=0.05 are fixed. Sensitivity is unknown.
- **How:** Grid search + Sobol sensitivity indices. Propose adaptive heuristics.
- **Venues:** J.UCS (natural follow-up), Applied Soft Computing

---

## Recommended Prioritization

```
Months 1-6 (near-term):
├── Direction 1: Multi-Class DSGD++  ← prerequisite for most other work
├── Direction 3: Rule Mining Integration  ← promised in DSGD++ paper
├── Direction 2: Conformal Prediction Wrapper  ← high novelty + feasibility
└── Direction 15: Hyperparameter Sensitivity  ← quick win, strengthens DSGD++

Months 6-12 (medium-term):
├── Direction 5: Convergence Theory  ← theoretical depth, top venues
├── Direction 6: Federated DSGD++  ← strongest narrative fit
├── Direction 4: UQ Benchmark  ← positions DSGD++ in landscape
└── Direction 8: Active Learning  ← practical + builds on D1

Months 12+ (long-term):
├── Direction 9: Non-Singleton Focal Elements
├── Direction 10: Deep Learning + DSGD++ Hybrid
└── Direction 11: Online/Streaming DSGD++
```

---

## 5. Full Reference List

### DSGD Lineage
1. Peñafiel, S., Baloian, N., Sanson, H., Pino, J.A. (2020). "Applying DST for Developing a Flexible, Accurate and Interpretable Classifier." *Expert Systems with Applications*, 148, 113262.
2. Peñafiel, S. (2021). "Interpretable Method for General Classification Using DST." *Master's Thesis, Universidad de Chile.*
3. Baloian, N., Davtyan, E., Petrosyan, K., Poghosyan, A., Harutyunyan, A., Penafiel, S. (2024). "Embedded Interpretable Regression using DST." *CODASSCA.*
4. Baloyan, A., et al. (2024). "Empirical Analysis of Feature Engineering for DS Classifier." *CODASSCA.*
5. Valdivia, R., et al. (2024/2025). "Interpretable Clustering Using DST." *CODASSCA / J.UCS.*
6. Tarkhanyan, A., Harutyunyan, A. (2025). "DSGD++." *J.UCS*, 31(9), 1004-1014. DOI: 10.3897/jucs.164745
7. Tarkhanyan, A., Harutyunyan, A. (2024). "Improving DSGD with MAF Initialization." *CODASSCA*, 137-142.
8. Sedláček, O., Bartoš, V. (2024). "Fusing Heterogeneous Data for Network Asset Classification." *IEEE NOMS.*

### Evidential Deep Learning
9. Sensoy, M., Kaplan, L., Kandemir, M. (2018). "Evidential Deep Learning to Quantify Classification Uncertainty." *NeurIPS.*
10. Amini, A., et al. (2020). "Deep Evidential Regression." *NeurIPS.*
11. Charpentier, B., et al. (2020). "Posterior Network." *NeurIPS.*
12. Tong, Z., Xu, P., Denoeux, T. (2021). "Evidential Classifier Based on DST and Deep Learning." *Neurocomputing*, 450, 275-293.
13. Kandemir, M., et al. (2024). "Are EDL Uncertainty Capabilities a Mirage?" *NeurIPS.*
14. Chen, et al. (2024). "Comprehensive Survey on EDL." *arXiv / IEEE TPAMI.*

### Uncertainty Measures
15. Deng, Y. (2016). "Deng Entropy." *Chaos, Solitons & Fractals*, 91, 549-553.
16. Pan, L., Deng, Y. (2023). "Belief interval-based total uncertainty measure for DST." *Information Sciences.*
17. Jirousek, R., Shenoy, P. (2023). "Numerical comparison of uncertainty measures." *IJAR.*
18. Chen, Y., Deng, Y. (2024). "Complex Deng Entropy." *Engineering Applications of AI.*

### Rule Mining + Belief Rule Bases
19. Yang, J.-B., et al. (2006). "Belief Rule-Base Inference Methodology (RIMER)." *IEEE Trans. SMC-A*, 36(2), 266-285.
20. Zhang, A., et al. (2020). "Rule Reduction for Extended BRB Based on DBSCAN." *IJAR*, 119, 20-39.
21. Fumanal-Idocin, J., et al. (2025). "Compact Rule-Based Classifier via Gradient Descent." *arXiv:2502.01375.*
22. Wang, Y., et al. (2025). "Distilling Neural Knowledge into Interpretable Belief Rule Bases." *AAAI/OpenReview.*

### Conformal Prediction + DST
23. Hofer, M., Krempl, G. (2024). "DST + Conformal Prediction." *ECML-PKDD.*
24. Karimi, et al. (2024). "Evidential Uncertainty Sets via Conformal Prediction." *COPA (PMLR).*

### Active Learning
25. Horanyi, M., Hullermeier, E. (2024). "Evidential Uncertainty Sampling for Active Learning." *Machine Learning (Springer).*
26. Horanyi, M., Hullermeier, E. (2023). "DS-Inspired Uncertainty Measures for Active Learning." *BELIEF (LNCS).*

### Multi-Class Scalability
27. Denoeux, T. (1995). "EK-NN Classification Rule Based on DST." *IEEE Trans. SMC*, 25(5), 804-813.
28. Denoeux, T. (2000). "Neural Network Classifier Based on DST." *IEEE Trans. SMC-A*, 30(2), 131-150.
29. Denoeux, T., Masson, M.-H. (2012). "Evidential Reasoning in Large Partially Ordered Sets." *Annals of OR*, 195, 135-161.
30. Skau, E. (2023). "Open World DST Using Complementary Sets." *AISTATS (PMLR).*

### Combination Rules
31. Yager, R. (1987). "On the DST Framework and New Combination Rules." *Information Sciences*, 41(2), 93-137.
32. Murphy, C. (2000). "Combining Belief Functions When Evidence Conflicts." *Decision Support Systems*, 29(1), 1-9.
33. Lefevre, E., Colot, O., Vannoorenberghe, P. (2002). "Belief Functions Combination and Conflict Management." *Information Fusion*, 3(2), 149-162.

### Federated/Distributed
34. Li, W., et al. (2023). "Multisensor Data Fusion in IoT via DST." *Sensors*, 23(11), 5141.

### Initialization Analogues
35. Erber, N., Laor, A. (2025). "Laor Initialization: Weight Init via KMeans for Backpropagation." *Big Data and Cognitive Computing*, 9(7), 181.
36. Ash, J., Adams, R.P. (2020). "On Warm-Starting Neural Network Training." *NeurIPS.*

### Surveys
37. Zhao, Y., et al. (2022). "Survey on DST in Classification." *—*
38. Chen, Z., et al. (2023). "Belief Theory Meets Deep Learning." *Information Sciences.*
39. Hamdi, H., et al. (2023). "Survey on Belief Functions in ML." *—*
40. Cheng, C., et al. (2024). "Comprehensive Survey of BRB Hybrid Expert Systems." *arXiv:2402.16651.*

### Interpretable ML
41. Letham, B., et al. (2015). "Interpretable Classifiers Using Rules and Bayesian Analysis." *Annals of Applied Statistics*, 9(3), 1350-1371.
42. Benard, C., et al. (2021). "SIRUS: Stable and Interpretable Rule Sets from Random Forests." *AISTATS (PMLR).*
43. Singh, C., et al. (2021). "imodels: Interpretable Models Python Package." *JOSS*, 6(61), 3192.

---

*This roadmap synthesizes findings from 90+ papers. Detailed per-theme reviews are available in the companion files: `literature_review_DST_classification.md`, `dst_modern_ml_literature_survey.md`, and `literature_review_systematic.md`.*
