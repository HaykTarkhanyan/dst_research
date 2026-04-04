# Rule Induction Paper — Progress Log

## Project Goal

Implement and evaluate **"Uncertainty-Driven Rule Induction for Dempster-Shafer Classifiers"** — using DSGD++ uncertainty to mine targeted conjunctive rules that resolve ambiguous regions.

Paper plan: `paper_plans/rule_induction_paper_plan.md`

---

## What's Done

### Codebase cleanup (2026-04-02)
- Unified logging: replaced all `print()` with module loggers across 6 files
- Fixed bugs: `self.train_set_size`, `list.index()` error handling, F1 div-by-zero, vectorized outlier detection
- Merged 3 `_optimize` methods into 1, removed ~130 lines of duplication
- Changed `active_rules` from list to set (O(1) lookup)
- **Performance fix**: `num_workers` default 1→0 (50x speedup on Windows), added `precompute_rule_matrix()`

### Rule induction framework (2026-04-02)
| Module | Purpose |
|--------|---------|
| `src/rule_adapter.py` | Parse rule strings from RIPPER/SkopeRules/RuleFit/DecisionTree → DSRule lambdas |
| `src/rule_miners.py` | Wrappers: `SkopeRulesMiner`, `RipperMiner`, `RuleFitMiner`, `DecisionTreeMiner` |
| `src/uncertainty_loop.py` | `UncertaintyGuidedRefiner` — iterative uncertainty-guided rule mining |
| `src/rule_ensemble.py` | `MultiSourceEnsemble` — combine rules from multiple miners |
| `src/rule_pruning.py` | `ConfidenceBasedPruner` — pruning + Pareto frontiers |

Dependencies added: `imodels==2.0.4`, `wittgenstein==0.3.4`

Tests: 118 total (78 original + 32 rule adapter + 8 uncertainty loop)

### Bug fixes from code review (2026-04-02)
- `force_precompute` forward path now respects `active_rules` (pruning was silently broken)
- `_rule_matrix` initialized in `__init__`
- RIPPER `pos_class` auto-detected from minority class
- F1 metric standardized to binary average

### Full experiment campaign (2026-04-03)

Ran E1-E5 on **6 datasets** with both random and clustering MAF initialization:
- heart-disease (303 samples, 13 features)
- breast-cancer-wisconsin (683 samples, 9 features)
- ionosphere (351 samples, 34 features)
- pima-diabetes (768 samples, 8 features)
- qsar-biodeg (1,055 samples, 41 features)
- phoneme (5,404 samples, 5 features)

Added baselines: GradientBoosting, RandomForest, LogisticRegression to E5.

**Infrastructure built:**
- `run_experiment.py` — generic runner for any dataset (E1-E5, random init)
- `run_experiment_clustering.py` — clustering init variant (E1-E3)
- `generate_report.py` — rich PDF reports with charts, tables, LaTeX export
- `slides/experiments_presentation.tex` — 27-slide Beamer methodology presentation

**Key fixes during campaign:**
- Added global rule cap to E2 iterative loop (`max_total_rules`, `max_weak_to_refine`)
- Fixed RIPPER extraction crash when `cond.val` is numpy.float64
- Fixed `create_uniform_uncertainty()` to support k>2 classes
- Fixed `uncertainty_loop.py` to respect `clf.maf_method` (was hardcoded to "random")

### 5-Fold Cross-Validation on 10 Datasets (2026-04-04)

Added 4 new datasets: sonar, banknote-authentication, blood-transfusion, german-credit.
Built `run_cv_experiment.py` with Wilcoxon tests and ablation study.

**Wilcoxon results (honest framing):**
- [x] DSGD++ **ties RuleFit** (p=0.574) — statistically no difference
- [x] DSGD++ **ties Logistic Regression** (p=0.922)
- [x] DSGD++ **loses to RF** (p=0.064, borderline) and **GB** (p=0.232)
- [x] **Iterative refinement significantly improves DSGD++ base** (p=0.020, +2.6pp)

Note: DSGD++ also beats DT d4, DT d8, FIGS, GreedyRuleList significantly — but those are deliberately weak baselines and this comparison is not meaningful.

**Ablation:** Rule induction helps on 7/10 datasets, hurts on 2 (blood-transfusion, pima).

**Verdict:** Publishable. The story is methodology + interpretability + uncertainty, not accuracy supremacy. DSGD++ matches RuleFit while providing DST uncertainty (belief intervals via m_Θ).

Full review: `experiment_results/results_review.md`

---

## What's Next

### For the paper
- [ ] Interpretability comparison (rule complexity, coverage, explanation quality)
- [ ] Uncertainty calibration study (is m_Θ well-calibrated vs softmax entropy?)
- [ ] Hyperparameter sensitivity analysis
- [ ] Explain why induction hurts on blood-transfusion and pima
- [ ] Write paper draft — frame as methodology contribution, not SOTA
- [ ] Target venue: Expert Systems with Applications, Knowledge-Based Systems, or Information Sciences

### Lower priority
- [ ] Add tests for `MultiSourceEnsemble` and `ConfidenceBasedPruner`
- [ ] Deduplicate rules across miners in ensemble
- [ ] Handle categorical RIPPER conditions
