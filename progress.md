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

### Heart Disease Experiment (2026-04-02/03)
Ran E1-E5 on heart-statlog dataset (270 samples, 13 features, binary).

**Results:**

| Experiment | Method | Acc | F1 | AUC | Rules |
|------------|--------|-----|-----|-----|-------|
| E1 | DSGD++ baseline (single-feature) | 0.654 | 0.576 | 0.693 | 38 |
| **E1** | **DSGD++ + SkopeRules** | **0.778** | **0.769** | **0.861** | **167** |
| E1 | DSGD++ + DecisionTree | 0.679 | 0.683 | 0.678 | 51 |
| E1 | DSGD++ + RIPPER | 0.642 | 0.554 | 0.704 | 38 |
| E2 | Iterative (SkopeRules) | 0.543 | 0.000 | 0.832 | 1049 |
| E2 | Iterative (DecisionTree) | 0.778 | 0.735 | 0.832 | 685 |
| E3 | Multi-source ensemble | 0.741 | 0.741 | 0.817 | 55 |
| E5 | RuleFit (standalone baseline) | 0.815 | 0.783 | 0.891 | - |
| E5 | FIGS (standalone baseline) | 0.802 | 0.784 | 0.840 | - |
| E5 | RIPPER (standalone baseline) | 0.802 | 0.778 | N/A | - |
| E5 | DecisionTree d4 (baseline) | 0.765 | 0.765 | 0.827 | - |

**Key findings:**
1. SkopeRules augmentation gives +12.4pp accuracy, +16.8pp AUC over baseline
2. DSGD++ + SkopeRules competitive with best interpretable baselines (RuleFit, FIGS) while providing DST uncertainty
3. Multi-source ensemble (55 rules) gives good accuracy/interpretability tradeoff
4. Iterative loop (E2) overfits with SkopeRules — needs global rule cap
5. RIPPER produces 0 rules on standardized data
6. Confidence-based pruning degrades too fast; random outperforms it

Rules + uncertainty scores saved in `experiment_results/heart-disease/rules/`

---

## What's Next

### Immediate (fix known issues)
- [ ] Add global cap on total mined rules in E2 iterative loop (e.g., max 30 total)
- [ ] Chain E2 (refine) + E4 (prune) — get accuracy benefit with controlled rule count
- [ ] Feed RIPPER unscaled/discretized data
- [ ] Improve pruning criterion (try coverage-based or hybrid)
- [ ] Vectorize `precompute_rule_matrix` with numba for larger datasets

### Experiments to run
- [ ] Run on breast-cancer-wisconsin (direct comparison with published DSGD++ results)
- [ ] Run on other datasets: brain tumor, PIMA diabetes, gaussian/rectangle/uniform
- [ ] E6: Scaling test on adult (48K rows) and bank-full (45K rows)
- [ ] Cross-validation (current results are single train/test split)

### Paper writing
- [ ] Figures: loss curves, Pareto frontier, uncertainty per iteration
- [ ] Interpretability case study: show actual rules with masses on heart disease
- [ ] Comparison table for paper (E1 + E5 combined)
- [ ] Write paper draft (target: Expert Systems with Applications or Information Fusion)

### Lower priority
- [ ] Add tests for `MultiSourceEnsemble` and `ConfidenceBasedPruner`
- [ ] Add tests for SkopeRules/RuleFit/RIPPER extraction (mock-based)
- [ ] Deduplicate rules across miners in ensemble
- [ ] Handle categorical RIPPER conditions (currently dropped)
