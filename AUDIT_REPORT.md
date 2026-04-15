# DSGD++ Codebase Audit Report

**Date:** 2026-04-03
**Scope:** Full codebase audit + paper plans critique

---

## Part 1: Critical Bugs in Core Implementation

### BUG-1: Broken PyTorch Parameter Registration (`DSModelMultiQ.py`)
**Severity: CRITICAL**

Parameters are created with `torch.tensor(..., requires_grad=True)` instead of `nn.Parameter()`. This bypasses PyTorch's module system:
- `.to(device)` won't move parameters to GPU
- `state_dict()` / `load_state_dict()` won't include them
- Standard optimizers may not see all parameters correctly
- The custom `parameters()` override returns a raw list instead of a generator

**Fix:** Replace `torch.tensor(..., requires_grad=True)` with `nn.Parameter(torch.tensor(...))` and register them properly.

---

### BUG-2: Data Leakage via Row Index Injection (`DSClassifierMultiQ.py`)
**Severity: CRITICAL**

```python
np.insert(X, 0, values=np.arange(0, len(X)), axis=1)
```

This prepends row indices as feature column 0. All rules referencing `x[0]` now check the row index instead of the first real feature. This silently corrupts the entire model — rules fire based on sample ordering rather than feature values.

---

### BUG-3: Division-by-Zero in Core DST Math (`core.py`)
**Severity: HIGH**

Multiple locations use floating-point equality (`== 0`) instead of tolerance-based comparisons:
- `dempster_rule()` — conflict normalization
- `cls_score()` — score computation
- `cls_pla_score()` — plausibility computation

**Fix:** Use `abs(x) < epsilon` guards.

---

### BUG-4: Broken Pickle Serialization (`DSModel.py`)
**Severity: HIGH**

Uses text mode (`"r"`/`"w"`) instead of binary mode (`"rb"`/`"wb"`) for pickle I/O. These calls crash at runtime.

---

### BUG-5: GPU Crash in predict() (`DSClassifierMultiQ.py:253`)
**Severity: HIGH**

Calls `.numpy()` without `.cpu()` first. Crashes on any CUDA-enabled system.

---

### BUG-6: Hardcoded `/3` in `dempster_rule_kt()` (`core.py`)
**Severity: HIGH**

Should depend on `k` (number of classes). Produces wrong results for any non-3-class problem.

---

### BUG-7: Silent normalize() Failure (`DSModel.py`)
**Severity: HIGH**

`normalize()` assigns to a local variable instead of updating `self.masses[i]`. Normalization silently does nothing.

---

### BUG-8: Possibly Incorrect Plausibility Formula (`core.py:59`)
**Severity: MEDIUM**

`1 - m[2 - cls]` may use wrong index for computing plausibility, especially in multi-class settings.

---

### BUG-9: Hardcoded `n_clusters=2` in KMeans (`DSEnhanced.py`)
**Severity: MEDIUM**

KMeans always uses 2 clusters regardless of number of classes. Won't generalize to multi-class.

---

### BUG-10: Negative Index Bug in Tree Rule Extraction (`rule_adapter.py:285-287`)
**Severity: MEDIUM**

sklearn's tree uses -2 for leaf nodes. The bounds check `feat_idx < len(feature_names)` passes for -2 (Python negative indexing), silently mapping to the wrong feature.

**Fix:** Add `if feat_idx < 0: return`

---

## Part 2: Performance Issues

### PERF-1: O(n*m) Nested Loop in Coverage Computation (`rule_pruning.py:46-50`)

```python
for i in range(len(X_train)):
    for j in range(n_rules):
        if clf.model.preds[j](X_train[i]):
            coverage[j] += 1
```

With 100K samples and 200 rules = 20M rule evaluations. Should be vectorized.

### PERF-2: Same Issue in Uncertainty Loop (`uncertainty_loop.py:133`)

```python
covered_mask = np.array([bool(pred(X_train[i])) for i in range(len(X_train))])
```

Called for each weak rule, each iteration.

### PERF-3: Deep Copy in `pareto_frontier()` (`rule_pruning.py:109`)

`copy.deepcopy(clf)` creates full copies of PyTorch models for each threshold (20 thresholds = 20 full model copies). Should save/restore state instead.

### PERF-4: Numerical Instability (`DSModelMultiQ.py`)

Adding `1e-16` is insufficient. `check_nan()` method exists but is never called during training.

---

## Part 3: Code Quality Issues

### QUAL-1: No fit() Guard in Rule Miners (`rule_miners.py`)
All miners access `self.clf_` in `extract_rules()` without checking if `fit()` was called.

### QUAL-2: Overly Broad Exception Handling (multiple files)
Catching bare `Exception` masks real bugs (TypeError, AttributeError) in `rule_adapter.py:252`, `uncertainty_loop.py:150`, `rule_ensemble.py:86`.

### QUAL-3: Division-by-Zero in Density Normalization (`utils.py:168`)
Divides by `(max - min)` without checking equality. NaN check comes AFTER the division.

### QUAL-4: 50+ Lines of Unverified Manual Gradient Code (`core.py`)
Unmaintainable hand-computed gradients with no verification against autograd.

### QUAL-5: DataFrame Mutation in Pipeline (`DSEnhanced.py`)
`labels_clustering` column added to mutable DataFrame causes issues on re-runs.

### QUAL-6: Missing Input Validation
- No X/y shape mismatch checks
- No column_names length validation
- No empty dataset handling

---

## Part 4: Test Suite Issues

### 8 Untested Functions in `utils.py`
- `get_distance()` — Core distance calculation
- `filter_by_rule()` — 79 lines of complex logic
- `run_dbscan()` — DBSCAN parameter search
- `remove_outliers_and_normalize()` — Normalization pipeline
- `dbscan_predict()` — DBSCAN prediction
- `evaluate_clustering()` — Clustering metrics
- `report_results()` — Results reporting
- `get_kdist_plot()` — Visualization

### Weak Assertions
- `test_predict_valid_classes()` passes if model predicts all class 0
- `test_predict_shape()` only checks shape, not values
- `test_generated_rules_fire_correctly()` uses 50% threshold (extremely weak)
- `test_cls_score_range()` only checks bounds [0,1], not correctness
- Probability sum tolerance of 1e-4 too loose (should be 1e-6)

### Flakiness Risk
- Module-scoped fixtures cause test coupling
- Missing `torch.manual_seed()` — inconsistent across CUDA/CPU
- No conftest.py seed management for torch

### Dependencies
- `wittgenstein==0.3.4` potentially unmaintained (last release ~2019)
- No dev dependencies (pytest-cov, pytest-mock)

---

## Part 5: Paper Plans Critique

### Plan 1: FedDSGD — Federated DSGD++

**Novelty: OVERSTATED**
- Denoeux (2021), "Distributed combination of belief functions" in Information Fusion directly addresses combining belief functions across distributed agents via Dempster's rule
- FL-IRT (2024, IEEE) addresses federated learning for interpretable rule-based systems
- Federated decision tree aggregation (2024, Information Sciences) covers interpretable model aggregation

**Critical Flaws:**
1. Federated clients are NOT independent — violates Dempster's rule assumption
2. "Consistency proof" via associativity/commutativity is mathematically incorrect (gives order-invariance, not convergence to centralized solution)
3. Communication efficiency comparison vs. MLPs is misleading (compare vs. other interpretable models)
4. Privacy claims are naive — MAFs encode class distributions

**Verdict:** Core novelty claim invalidated by Denoeux (2021). Reframe as empirical comparison study.

### Plan 2: GA-DSGD — Evolutionary Rule Optimization

**Novelty: LARGELY REDUNDANT**
- BRB evolutionary optimization: 400+ papers surveyed in arXiv:2402.16651
- HEROS (GECCO 2025) does exactly the Pareto frontier idea, already open-sourced
- "DST-specific fitness" is structurally identical to precision-recall-diversity

**Feasibility Crisis:**
- 2,500-40,000 DSGD++ training runs vs. one RIPPER call
- Most novel idea (evolving frame of discernment) only helps with 6+ classes; DSGD++ only tested on binary

**Verdict:** Low novelty, low feasibility. Weakest of the three plans.

### Plan 3: Rule Induction — Uncertainty-Driven Mining

**Novelty: MODERATE (strongest of three)**
- Core loop resembles active learning/boosting but specific DST formulation is reasonably novel

**Critical Flaws:**
1. Convergence not guaranteed — new rules change entire model via Dempster combination
2. Missing baselines: FRR (2025, 5% over RIPPER without DST) and Rule Distillation (AAAI 2026)
3. No statistical methodology (significance tests, cross-validation, confidence intervals)
4. Never tests whether DST uncertainty > standard probabilistic entropy

**Verdict:** Most viable plan if properly scoped and honest about novelty.

### Cross-Cutting Issues
- DSGD++ unproven at scale (only tested on 699-3762 samples, binary)
- Zadeh's counterexample for Dempster's rule ignored in all plans
- Self-citation loop — missing critical recent work
- Unrealistic timelines ("1 week to write paper")

### Plan Rankings

| Plan | Novelty | Feasibility | Impact | Recommendation |
|------|---------|-------------|--------|----------------|
| Rule Induction | Moderate | High | Moderate | Pursue (Idea 1 only), add proper baselines |
| Federated | Low-Moderate | Moderate | Moderate | Reframe as empirical study |
| GA Optimization | Low | Low | Low | Deprioritize |

---

## Key References Missing from Paper Plans

1. Denoeux (2021) — "Distributed combination of belief functions," Information Fusion
2. HEROS (GECCO 2025) — NSGA-II multi-objective rule-set optimization
3. FRR (Fumanal-Idocin 2025) — Gradient-based rule learning, 5% over RIPPER
4. Rule Distillation (AAAI 2026) — Neural knowledge into belief rule bases
5. BRB-EA Survey (2024, arXiv:2402.16651) — 400+ papers on evolutionary BRB optimization
6. PIBRIA (2024, ScienceDirect) — Belief rule classification with FURIA
