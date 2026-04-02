# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DSGD++ is a Dempster-Shafer Theory (DST) based classifier that uses gradient descent to optimize Mass Assignment Functions (MAFs). It generates interpretable rules from data, learns belief mass assignments via backpropagation, and combines evidence using Dempster's rule of combination.

## Repository Structure

- `src/` — DSGD++ implementation (Python/PyTorch)
  - `notebooks/` — Jupyter notebooks (recovered from DSGD-Enhanced repo)
    - `clean.ipynb` — end-to-end pipeline: data loading, scaling, clustering (KMeans/DBSCAN), distance calc, DST training
    - `use_this_sept_1.ipynb` — batch experiment runner across datasets, in-between rules ablation study
    - `compare_rules.ipynb` — rule comparison between MAF methods (clustering vs random), importance scoring, LaTeX export
    - `generate_report.ipynb` — ReportGenerator class: aggregates experiment results, uncertainty analysis, paper tables
    - `run_clustering.ipynb` — clustering pipeline with DBSCAN/KMeans, adjusted density calculation, outlier detection
    - `process_datasets.ipynb` — dataset preprocessing: adult, bank-full, german_credit — label encoding, missing values
    - `interpret_findings.ipynb` — MAF comparison analysis: accuracy, F1, training time, loss across datasets (J.UCS paper)
    - `bzbz.ipynb` — scratch/exploration notebook
- `paper/` — LaTeX source for the published paper
- `paper_plans/` — future research directions (federated, GA optimization, rule induction)
- `lit_review/` — literature review documents

## Setup

```bash
pip install -r requirements.txt
```

Core dependencies: PyTorch 1.12, scikit-learn 1.2, numpy, scipy, pandas, plotly, dill.

## Architecture

### Class Hierarchy

```
core.py (DST math: MAFs, Dempster's rule, belief/plausibility)
  └─ DSRule.py (predicate + caption wrapper)
  └─ DSModelMultiQ.py(nn.Module) — k-class model, stores rules + learnable mass parameters
      └─ DSClassifierMultiQ.py(ClassifierMixin) — sklearn-compatible trainer (fit/predict/predict_proba)
          └─ DSEnhanced.py — end-to-end pipeline orchestrator (data→clustering→rules→train→evaluate)
```

`DSModel.py` is the older binary-only version; `DSModelMultiQ.py` is the current general k-class model.

### Key Abstractions

**Rules** (`DSRule`): Boolean predicates with captions, e.g. `DSRule(lambda x: x[0] > 5, "Feature0 > 5")`. Generated automatically from statistical breaks (quantiles) per feature, with optional pairwise multiplication rules.

**Mass Assignment Functions (MAFs)**: Vectors `(m_class1, ..., m_classK, m_uncertainty)` attached to each rule. Three initialization methods:
- `random` — small random belief, high uncertainty; fully learnable
- `clustering` — derived from cluster distances and class proportions
- `uniform` — equal belief across classes

**Forward pass**: For input x, select rules where predicate(x)=True → convert masses to commonalities (q = m_singletons + m_uncertainty) → multiply element-wise → normalize to probabilities.

**Training**: Gradient descent (Adam/SGD) on mass parameters with MSE or cross-entropy loss. `normalize()` is called after every optimizer step to maintain DST constraints (masses in [0,1], sum ≤ 1).

### Pipeline (DSEnhanced)

`read_data → preprocess_data → train_test_split → standard_scaling → clustering → train_DST → evaluate → save_results`

### Running Experiments

```python
from src.DSEnhanced import DSEnhanced

ds = DSEnhanced(dataset="breast-cancer-wisconsin", clustering_method="kmeans",
                maf_method="clustering", breaks=3)
ds.read_data()
ds.preprocess_data()
ds.train_test_split()
ds.standard_scaling()
ds.clustering_and_inference()
ds.train_DST()
```

Or use `DSClassifierMultiQ` directly for sklearn-style usage:

```python
from src.DSClassifierMultiQ import DSClassifierMultiQ

clf = DSClassifierMultiQ(num_classes=2, lr=0.005, max_iter=500,
                         maf_method="random", lossfn="CE")
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
clf.predict_explain(X_test[0])  # explainability: shows fired rules and masses
```

## Tests

Run the test suite with:

```bash
python -m pytest tests/ -v
```

78 tests across 4 files covering `core.py`, `DSRule.py`, `DSModelMultiQ.py`, `DSClassifierMultiQ.py`, and `utils.py`. Tests use minimal datasets (8 samples/class, 3 epochs) and module-scoped fixtures to keep runtime around 3 minutes. All tests use seed 509.
