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

### Results summary (2026-04-03)

| Dataset | Baseline | Best DSGD++ | vs Interpretable | vs Black-box |
|---------|----------|-------------|-----------------|--------------|
| heart-disease | 65.9% | 78.0% (+12.1pp) | LOSE (RuleFit 81.5%) | LOSE |
| breast-cancer | 97.1% | 98.1% (+1.0pp) | WIN | TIE |
| ionosphere | 91.5% | 95.3% (+3.8pp) | **WIN** | **WIN** |
| pima-diabetes | 73.6% | 73.6% (+0.0pp) | LOSE (RuleFit 75.3%) | LOSE |
| qsar-biodeg | 82.7% | 87.4% (+4.7pp) | **WIN** | **WIN** |
| phoneme | 77.8% | 83.5% (+5.7pp) | LOSE (DT d8 86.1%) | LOSE |

**Verdict:** 2 wins, 3 losses, 1 tie vs interpretable baselines. Promising but not conclusive.

Full review: `experiment_results/results_review.md`

---

## What's Next

### High priority (needed for publication)
- [ ] **5-fold cross-validation** on all 6 datasets with standard deviations
- [ ] **Wilcoxon signed-rank test** across datasets
- [ ] **Add 4-5 more datasets** (adult, german-credit, sonar, banknote, blood-transfusion)
- [ ] **Ablation study** (base → +mining → +iterative → +ensemble)

### Medium priority (strengthens the paper)
- [ ] Interpretability comparison (avg conditions/rule, coverage overlap)
- [ ] Hyperparameter sensitivity (threshold, cap, iterations)
- [ ] Uncertainty calibration comparison (DSGD++ m_Theta vs softmax entropy)
- [ ] Explain why PIMA shows zero improvement

### Paper writing
- [ ] Frame contribution as methodology (uncertainty loop), not "beating RuleFit"
- [ ] Lean into interpretability + uncertainty angle
- [ ] Target venue: Expert Systems with Applications or Information Fusion
- [ ] Write paper draft

### Lower priority
- [ ] Add tests for `MultiSourceEnsemble` and `ConfidenceBasedPruner`
- [ ] Deduplicate rules across miners in ensemble
- [ ] Handle categorical RIPPER conditions
