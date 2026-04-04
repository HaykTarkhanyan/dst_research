# Experimental Results Review — April 2026

## Status: Publishable with the right framing. Not a "we beat everything" paper.

---

## 5-Fold Cross-Validation Results (10 datasets)

| Dataset | DSGD++ base | DSGD++ best | Best method | RuleFit | RF |
|---------|------------|------------|-------------|---------|-----|
| heart-disease | .600±.072 | **.793±.054** | ensemble | .830±.030 | .811±.032 |
| breast-cancer | .972±.007 | **.975±.006** | iterative | .962±.007 | .972±.010 |
| ionosphere | .897±.010 | .926±.034 | iterative | .931±.023 | .937±.028 |
| pima-diabetes | .760±.024 | .760±.024 | base | .759±.035 | .755±.019 |
| qsar-biodeg | .819±.037 | .833±.024 | augmented | .846±.006 | .869±.013 |
| phoneme | .784±.013 | .806±.005 | iterative | .811±.012 | .880±.009 |
| sonar | .793±.087 | .798±.068 | ensemble | .788±.058 | .841±.049 |
| banknote | .887±.043 | **.981±.009** | ensemble | .978±.016 | .994±.006 |
| blood-transfusion | **.779±.014** | .779±.014 | base | .771±.008 | .755±.014 |
| german-credit | .620±.088 | .700±.019 | iterative | .725±.042 | .749±.027 |

### vs RuleFit: 5 wins, 5 losses (statistical tie, p=0.574)
### vs RandomForest: 3 wins, 7 losses (p=0.064, borderline)
### vs LogisticRegression: 5 wins, 4 losses, 1 tie (p=0.922, no difference)

---

## Wilcoxon Signed-Rank Tests

| Comparison | Mean diff | p-value | Verdict |
|-----------|-----------|---------|---------|
| DSGD++ vs RuleFit | -0.005 | 0.574 | **Tied** — no significant difference |
| DSGD++ vs LogisticRegression | +0.004 | 0.922 | **Tied** — no significant difference |
| DSGD++ vs RandomForest | -0.021 | 0.064 | DSGD++ worse (borderline) |
| DSGD++ vs GradientBoosting | -0.014 | 0.232 | DSGD++ worse (not significant) |
| **Iterative vs Base (ablation)** | **+0.026** | **0.020** | **Significant improvement** |

**Honest reading:** DSGD++ with rule induction is statistically equivalent to RuleFit and Logistic Regression. It loses to ensemble methods (RF, GB) as expected — those are much more complex models. The win over weak baselines (DT d4, FIGS, GreedyRuleList) is not meaningful since those are deliberately simple.

---

## Ablation Study

| Dataset | Base | +DT mining | +Iterative | +Ensemble |
|---------|------|-----------|------------|-----------|
| heart-disease | .600 | .719 (+.119) | .689 | **.793** (+.193) |
| breast-cancer | .972 | .971 | **.975** (+.003) | .968 |
| ionosphere | .897 | .914 | **.926** (+.029) | .920 |
| pima-diabetes | **.760** | .758 | .760 | .727 |
| qsar-biodeg | .819 | **.833** (+.014) | .829 | .828 |
| phoneme | .784 | .801 | **.806** (+.022) | .803 |
| sonar | .793 | .788 | .788 | **.798** (+.005) |
| banknote | .887 | .960 | .921 | **.981** (+.094) |
| blood-transfusion | **.779** | .759 | .775 | .765 |
| german-credit | .620 | .640 | **.700** (+.080) | .696 |

**Rule induction improves DSGD++ on 7/10 datasets** (avg +2.6pp, p=0.020).
Hurts on 2/10 (blood-transfusion, pima-diabetes — both small, few features).

---

## What the paper can honestly claim

1. **Rule induction significantly improves DSGD++** (p=0.020, +2.6pp avg) — this IS the contribution
2. **DSGD++ with rule induction matches RuleFit** (p=0.574) while providing explicit uncertainty quantification via m_Θ that RuleFit cannot
3. **The uncertainty-guided mining loop is principled** — uses the model's own signal to target rule generation
4. **Ablation shows each component contributes** — largest gains from ensemble (+9.4pp banknote) and iterative (+8pp german-credit, +19pp heart-disease)

## What the paper CANNOT claim

1. ~~"Beats all interpretable baselines"~~ — ties RuleFit, ties LR
2. ~~"Beats 4 baselines significantly"~~ — those 4 (DT d4, DT d8, FIGS, GRL) are deliberately weak; beating them is expected
3. ~~"Competitive with black-box models"~~ — loses to RF and GB consistently
4. ~~"Rule induction always helps"~~ — hurts on 2/10 datasets

## Paper framing recommendation

**Title angle:** "Uncertainty-Driven Rule Induction for Dempster-Shafer Classifiers"

**Core narrative:** Not about accuracy supremacy. About a principled methodology that:
- Uses DST uncertainty as a feedback signal for targeted rule mining
- Achieves accuracy on par with the best interpretable baselines (RuleFit)
- Additionally provides calibrated uncertainty (belief intervals via Bel/Pl)
- Shows clear ablation from base → augmented → iterative → ensemble

**Target venue:** Journal where methodology + interpretability matters more than SOTA numbers (e.g., Expert Systems with Applications, Knowledge-Based Systems, Information Sciences)

---

## Remaining gaps

- [ ] Interpretability comparison (rule complexity, coverage, explanation quality)
- [ ] Uncertainty calibration study (is m_Θ actually well-calibrated?)
- [ ] Hyperparameter sensitivity analysis
- [ ] Why does induction hurt on blood-transfusion and pima?

---

## Files

- `run_cv_experiment.py` — 5-fold CV with Wilcoxon + ablation
- `experiment_results/cv_results/` — all CV results (10 datasets)
- `experiment_results/cv_results/wilcoxon_tests.json` — statistical tests
- `experiment_results/cv_results/ablation.json` — ablation data
- `run_experiment.py` / `run_experiment_clustering.py` — single-split experiments
- `slides/experiments_presentation.tex` — methodology slides

*Review date: 2026-04-04 (updated with 5-fold CV on 10 datasets)*
