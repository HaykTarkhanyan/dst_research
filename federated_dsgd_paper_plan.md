# FedDSGD: Paper Plan for Federated Interpretable Classification via Dempster-Shafer Evidence Fusion

**Date:** April 1, 2026

---

## The Core Challenge

For a federated DSGD++ paper, you need to answer **one central question**:

> *Is Dempster's combination rule a better aggregation mechanism for federated interpretable classification than standard weight averaging?*

If yes, you have a paper.

---

## Paper Structure

### Title (working)
*"FedDSGD: Federated Interpretable Classification via Dempster-Shafer Evidence Fusion"*

### 1. Introduction (the narrative)

The story writes itself:

- Federated learning needs aggregation. FedAvg averages weights — a heuristic with no theoretical justification for combining *evidence*.
- Dempster's combination rule was literally invented to combine evidence from independent sources.
- Yet nobody has connected these two ideas for classification.
- DSGD++ provides the bridge: it's a classifier whose internal representation (MAFs) can be directly combined via Dempster's rule.

### 2. Problem Formulation

This is where you define the setup mathematically:

- $K$ clients, each with local dataset $D_k$
- All clients share the same feature space but different data distributions (non-IID)
- Goal: learn a global DSGD model without sharing raw data

### 3. Algorithm — this is the core contribution

There are **two design choices** that define the paper:

#### Design Choice A: What is shared across clients?

**Option 1 — Share MAFs only (rules are global)**
- All clients use the same rule set (e.g., generated from a small public seed dataset, or from feature statistics shared in advance)
- Each client optimizes MAFs locally on their data
- Server combines MAFs via Dempster's rule

This is the simpler version. The algorithm:

```
FedDSGD Algorithm:
1. INITIALIZATION
   - Server generates a global rule set R from feature statistics
     (e.g., "feature_i > threshold_j" for each feature)
   - Server sends R to all clients

2. LOCAL TRAINING (at each client k)
   - Compute representativeness on local data D_k (KMeans)
   - Initialize MAFs using DSGD++ confidence method
   - Optimize MAFs via gradient descent on D_k
   - Send optimized MAFs {m_k(r)} for each rule r to server

3. SERVER AGGREGATION
   - For each rule r, combine MAFs from all clients:
     m_global(r) = m_1(r) ⊕ m_2(r) ⊕ ... ⊕ m_K(r)
     where ⊕ is Dempster's combination rule
   - Send m_global back to all clients

4. Repeat steps 2-3 for T rounds
```

**Option 2 — Share rules AND MAFs (rules are local)**
- Each client generates its own rules from local data
- Server must align/merge rules + combine MAFs
- More complex but more powerful

Recommendation: start with **Option 1** — it's cleaner and the contribution is already clear.

#### Design Choice B: How to handle conflict?

When clients disagree strongly (e.g., Hospital A says mass(malignant)=0.9 for a rule, Hospital B says mass(benign)=0.8 for the same rule), Dempster's rule produces high conflict. You need a strategy:

- **Dempster's rule** (normalize conflict away) — the default
- **Murphy averaging** (average MAFs first, then combine) — reduces conflict sensitivity
- **Yager's rule** (assign conflict mass to full set = "I don't know") — more conservative
- **Weighted combination** (weight each client by data size or confidence) — practical

Testing all four and analyzing which works best under different non-IID settings is a natural experiment.

### 4. Theoretical Analysis

This gives the paper depth. You can prove:

**(a) Consistency:** If all clients have data from the same distribution, FedDSGD converges to the same model as centralized DSGD trained on all data combined. This follows from associativity and commutativity of Dempster's rule.

**(b) Communication efficiency:** Each client sends $|R| \times |C|$ floats (rules x classes), which is much smaller than neural network weight vectors. For a typical DSGD model: 30 rules x 2 classes = 60 floats per round vs. thousands for even a small neural network.

**(c) Interpretability preservation:** The global model is still a set of rules with MAFs — the same interpretable structure as local models.

### 5. Experiments

#### Datasets
Use existing datasets + standard ones, partitioned across clients:

| Dataset | Rows | Features | Classes | How to Partition |
|---------|------|----------|---------|-----------------|
| Breast Cancer Wisconsin | 699 | 9 | 2 | Split by hospital (simulate) |
| Heart Disease (Cleveland + Hungarian + Swiss) | 920 | 13 | 2 | **Natural federation** — already from different hospitals! |
| PIMA Diabetes | 768 | 8 | 2 | Random split across K clients |
| Adult Income (Census) | 48842 | 14 | 2 | Split by state/region |
| Gaussian/Rectangle/Uniform (from DSGD++) | 500-1263 | 3 | 2 | Random or biased splits |

The **Heart Disease** dataset is a gift — it's literally from different hospitals (Cleveland, Hungarian, Swiss, VA Long Beach). This is a *real* federated setting, not simulated.

#### Non-IID Settings
Test under increasing levels of data heterogeneity:
- **IID:** Uniform random split across clients
- **Quantity skew:** Clients have different amounts of data
- **Label skew:** Clients have different class ratios (e.g., Hospital A: 90% benign, Hospital B: 90% malignant)
- **Feature skew:** Clients have different feature distributions (e.g., different patient demographics)

#### Baselines
| Method | What it is |
|--------|-----------|
| **Centralized DSGD++** | Upper bound — all data pooled, no privacy |
| **Local-only DSGD++** | Lower bound — each client trains alone, no federation |
| **FedAvg + Logistic Regression** | Standard FL baseline on interpretable model |
| **FedAvg + Random Forest** | Standard FL baseline on tree model |
| **FedAvg + MLP** | Standard FL on a small neural network |
| **FedDSGD (ours)** | Dempster's combination for aggregation |

#### Metrics
| Metric | What it measures |
|--------|-----------------|
| Accuracy, F1 | Predictive performance |
| Communication cost (bytes/round) | How much data is sent |
| Rounds to converge | How fast federation works |
| Uncertainty calibration (ECE) | Are uncertainty scores reliable? |
| Rule count | Interpretability |

#### Key Experiments (these become tables/figures)

**Experiment 1:** Accuracy vs. number of clients (K=2, 5, 10, 20). Show that FedDSGD maintains accuracy as K grows.

**Experiment 2:** IID vs. non-IID performance degradation. Show that Dempster's rule handles non-IID better than FedAvg (because it models uncertainty explicitly rather than blindly averaging).

**Experiment 3:** Combination rule comparison (Dempster vs. Murphy vs. Yager vs. weighted). Table showing which rule works best under which non-IID setting.

**Experiment 4:** Communication efficiency. Bar chart showing bytes per round: FedDSGD (60 floats) vs. FedAvg-MLP (thousands of weights).

**Experiment 5:** Interpretability case study on Heart Disease. Show the global rules and MAFs learned from 4 hospitals. A clinician could read: "Rule: if chest_pain_type = asymptomatic AND max_heart_rate < 120 -> mass(disease) = 0.75. Evidence from 3/4 hospitals."

### 6. Discussion

Address:
- **Privacy:** MAFs are aggregated statistics, not raw data. But could an adversary reconstruct local data from MAFs? Likely no (MAFs are lossy summaries), but discuss.
- **Scalability:** Works well for tabular data with moderate rule counts. For 1000+ rules or many classes, Dempster's combination becomes expensive.
- **Limitation:** Currently binary classification (fixed if multi-class is done first).

---

## What Makes This Publishable

The paper has **four contributions**, any two of which would be sufficient:

1. **Novel algorithm** (FedDSGD) — first federated DST classifier
2. **Theoretical fit** — Dempster's rule was designed for exactly this
3. **Communication efficiency** — orders of magnitude less data sent than neural FL
4. **Interpretability** — global model remains human-readable (unique among FL methods)

---

## Realistic Assessment

| Aspect | Assessment |
|--------|-----------|
| **Novelty** | High — nobody has done this |
| **Implementation effort** | ~2-3 weeks for Option 1 (global rules) |
| **Biggest risk** | Performance under extreme non-IID may suffer if Dempster's rule produces high conflict |
| **Mitigation** | Test Murphy/Yager alternatives; this becomes an analysis contribution, not a weakness |
| **Best target venues** | Information Fusion, ECML-PKDD, IJCAI, or AAAI |

---

## Suggested Order of Work

```
Week 1: Implement FedDSGD (Option 1) on Breast Cancer + Heart Disease
Week 2: Run Experiments 1-2 (accuracy vs. K, IID vs. non-IID)
Week 3: Run Experiments 3-5 (combination rules, communication, case study)
Week 4: Write paper
```

**Dependency:** This is easier if multi-class DSGD++ is done first (Direction 1), but can be done with binary classification using the datasets above.

---

## Key References

1. McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS.* — FedAvg paper.
2. Shafer, G. (1976). "A Mathematical Theory of Evidence." *Princeton University Press.* — Dempster's combination rule.
3. Penafiel et al. (2020). "Applying DST for Developing a Flexible, Accurate and Interpretable Classifier." *Expert Systems with Applications.* — Original DSGD.
4. Tarkhanyan & Harutyunyan (2025). "DSGD++." *J.UCS.* — MAF initialization.
5. Murphy (2000). "Combining Belief Functions When Evidence Conflicts." *Decision Support Systems.* — Murphy averaging.
6. Yager (1987). "On the DST Framework and New Combination Rules." *Information Sciences.* — Yager's rule.
7. Li et al. (2023). "Multisensor Data Fusion in IoT via DST." *Sensors.* — DST for distributed fusion.
8. Chen et al. (2023). "Fusion Strategies for Federated Decision Making." *arXiv.* — Closest existing work.
