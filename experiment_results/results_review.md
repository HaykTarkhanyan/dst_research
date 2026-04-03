# Experimental Results Review — April 2026

## Status: Promising methodology, results not yet conclusive for a strong paper

---

## Scoreboard: DSGD++ best vs Baselines

| Dataset | DSGD++ base | DSGD++ best | Best method | RuleFit | RF | GB | vs Interp | vs BB |
|---------|------------|------------|-------------|---------|------|------|-----------|-------|
| heart-disease | 65.9% | 78.0% | E2 SkopeRules | 81.5% | 79.0% | 77.8% | LOSE | LOSE |
| breast-cancer | 97.1% | 98.1% | E3 Ensemble | 97.6% | 98.1% | 97.6% | WIN | TIE |
| ionosphere | 91.5% | 95.3% | E1 RIPPER | 91.5% | 90.6% | 91.5% | **WIN** | **WIN** |
| pima-diabetes | 73.6% | 73.6% | Baseline | 75.3% | 74.9% | 73.2% | LOSE | LOSE |
| qsar-biodeg | 82.7% | 87.4% | E2 SkopeRules | 83.3% | 85.8% | 84.5% | **WIN** | **WIN** |
| phoneme | 77.8% | 83.5% | E2 DT | 80.5% | 88.4% | 86.2% | LOSE | LOSE |

**vs interpretable baselines: 2 wins, 3 losses, 1 tie**
**vs black-box baselines: 2 wins, 3 losses, 1 tie**

---

## Rule Induction Improvement (the internal story)

| Dataset | Baseline | Best DSGD++ | Improvement |
|---------|----------|-------------|-------------|
| heart-disease | 65.9% | 78.0% | +12.1pp |
| phoneme | 77.8% | 83.5% | +5.7pp |
| qsar-biodeg | 82.7% | 87.4% | +4.7pp |
| ionosphere | 91.5% | 95.3% | +3.8pp |
| breast-cancer | 97.1% | 98.1% | +1.0pp |
| pima-diabetes | 73.6% | 73.6% | +0.0pp |

Rule induction improves DSGD++ on 5/6 datasets (avg +5.5pp on improved).

---

## Clustering vs Random Init (best per dataset)

| Dataset | Random best | Clustering best | Winner |
|---------|------------|----------------|--------|
| breast-cancer | 98.1% | 98.5% | Clustering (+0.4pp) |
| ionosphere | 95.3% | 93.4% | Random (+1.9pp) |
| phoneme | 83.5% | 84.4% | Clustering (+0.9pp) |
| qsar-biodeg | 87.4% | 87.1% | Tied |
| heart-disease | 78.0% | 77.8% | Tied |
| pima-diabetes | 73.6% | 73.6% | Tied |

**Takeaway:** Clustering gives a better starting point but gradient descent compensates. No consistent winner.

---

## Strengths

1. **Methodology is novel and principled** — uncertainty-guided iterative mining is a closed feedback loop
2. **Clear internal improvement** — 5/6 datasets show gains from rule induction
3. **Beats black-box on 2 datasets** while remaining fully interpretable (ionosphere, qsar-biodeg)
4. **Clean implementation** — 118 tests, 6 modules, reproducible pipeline
5. **Comprehensive evaluation** — 6 datasets x 5 experiments x 2 initializations x 10 baselines

## Weaknesses

1. **No cross-validation** — single 70/30 split, no error bars, results could be noise
2. **No statistical tests** — need Wilcoxon signed-rank across datasets
3. **Only 6 datasets** — 10-15 is the norm for rule-learning papers
4. **RuleFit is a tough competitor** — beats DSGD++ on 3/6 datasets
5. **No interpretability evaluation beyond rule count** — need complexity metrics or user study
6. **PIMA shows zero improvement** — need to explain why
7. **Heart disease (showcase) loses to RuleFit** — 78.0% vs 81.5%

---

## What's Needed for Publication

### Option A: Workshop / Short Paper (quickest path)
- [ ] 5-fold cross-validation on all 6 datasets with std devs
- [ ] Wilcoxon signed-rank test across datasets
- [ ] Frame contribution as methodology (uncertainty loop), not "beating RuleFit"
- [ ] Lean into interpretability + uncertainty angle

### Option B: Full Conference / Journal Paper
- [ ] Everything in Option A
- [ ] Add 4-5 more datasets (adult, german-credit, sonar, banknote, blood-transfusion)
- [ ] Ablation study (base → +mining → +iterative → +ensemble)
- [ ] Interpretability comparison (avg conditions/rule, coverage overlap)
- [ ] Hyperparameter sensitivity (threshold, cap, iterations)
- [ ] Uncertainty calibration comparison (DSGD++ m_Theta vs softmax entropy)

---

## Files and Artifacts

- `run_experiment.py` — generic experiment runner (E1-E5, random init)
- `run_experiment_clustering.py` — clustering-init variant (E1-E3)
- `generate_report.py` — PDF report generator with charts and LaTeX
- `generate_cross_dataset_report.py` — cross-dataset comparison report
- `slides/experiments_presentation.tex` — 27-slide Beamer methodology presentation
- `experiment_results/<dataset>/` — results for 6 datasets (random init)
- `experiment_results/<dataset>_clustering/` — results for 6 datasets (clustering init)
- All results include cached per-experiment JSONs, saved rules with uncertainty scores

---

*Review date: 2026-04-03*
