"""
5-fold cross-validation experiment runner with Wilcoxon test and ablation study.

Usage:
    python run_cv_experiment.py                    # all datasets
    python run_cv_experiment.py heart-disease       # single dataset
    python run_cv_experiment.py --folds 10          # 10-fold CV

Outputs:
    experiment_results/cv_results/<dataset>/fold_results.json
    experiment_results/cv_results/summary.json
    experiment_results/cv_results/wilcoxon_tests.json
    experiment_results/cv_results/ablation.json
"""
import argparse
import json
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.DSClassifierMultiQ import DSClassifierMultiQ
from src.rule_adapter import build_feature_name_map
from src.rule_miners import SkopeRulesMiner, RipperMiner, DecisionTreeMiner
from src.uncertainty_loop import UncertaintyGuidedRefiner
from src.rule_ensemble import MultiSourceEnsemble

warnings.filterwarnings("ignore")

SEED = 509
MAX_ITER = 300  # slightly reduced for CV speed


# ── Logging ──
def setup_logging():
    log = logging.getLogger("cv_experiment")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(ch)
    sys.stdout.reconfigure(line_buffering=True)
    return log


# ── Data ──
def load_dataset(name):
    path = os.path.join("src", "datasets", f"{name}.csv")
    df = pd.read_csv(path)
    y = df["labels"].values
    X = df.drop(columns=["labels"]).values
    cols = [c for c in df.columns if c != "labels"]
    return X, y, cols


# ── Evaluation ──
def evaluate(clf, X_test, y_test):
    preds = clf.predict(X_test)
    try:
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_test)
            auc = roc_auc_score(y_test, proba[:, 1])
        else:
            auc = None
    except Exception:
        auc = None
    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
    }


# ── DSGD++ Methods ──
def run_dsgd_base(X_tr, y_tr, X_te, y_te, cols):
    """Ablation step 1: Base single-feature rules only."""
    clf = DSClassifierMultiQ(num_classes=2, lr=0.005, max_iter=MAX_ITER,
                             lossfn="CE", precompute_rules=True, force_precompute=True)
    clf.fit(X_tr, y_tr, add_single_rules=True, single_rules_breaks=2, column_names=cols)
    return evaluate(clf, X_te, y_te)


def run_dsgd_augmented(X_tr, y_tr, X_te, y_te, cols):
    """Ablation step 2: Base + one-shot DecisionTree mining."""
    fm = build_feature_name_map(cols)
    clf = DSClassifierMultiQ(num_classes=2, lr=0.005, max_iter=MAX_ITER,
                             lossfn="CE", precompute_rules=True, force_precompute=True)
    clf.model.generate_statistic_single_rules(X_tr, breaks=2, column_names=cols)
    miner = DecisionTreeMiner(max_depth=4, random_state=SEED)
    try:
        miner.fit(X_tr, y_tr, feature_names=cols)
        for r in miner.extract_rules(fm):
            clf.model.add_rule(r, method="random")
    except Exception:
        pass
    clf.fit(X_tr, y_tr)
    return evaluate(clf, X_te, y_te)


def run_dsgd_iterative(X_tr, y_tr, X_te, y_te, cols):
    """Ablation step 3: Iterative uncertainty-guided refinement."""
    base_rules = len(cols) * 3
    total_cap = base_rules * 3
    clf = DSClassifierMultiQ(num_classes=2, lr=0.005, max_iter=MAX_ITER,
                             lossfn="CE", precompute_rules=True, force_precompute=True)
    miner = SkopeRulesMiner(n_estimators=50, precision_min=0.3, recall_min=0.01, max_rules=5)
    refiner = UncertaintyGuidedRefiner(
        clf, miner, uncertainty_threshold=0.3, max_iterations=3,
        min_covered_samples=10, max_rules_per_weak=3,
        max_total_rules=total_cap, max_weak_to_refine=10,
    )
    result = refiner.refine(X_tr, y_tr, column_names=cols, single_rules_breaks=2)
    return evaluate(result["clf"], X_te, y_te)


def run_dsgd_ensemble(X_tr, y_tr, X_te, y_te, cols):
    """Ablation step 4: Multi-source ensemble."""
    ensemble = MultiSourceEnsemble(
        miners=[
            SkopeRulesMiner(n_estimators=50, precision_min=0.3, recall_min=0.01, max_rules=10),
            RipperMiner(max_rules=10),
            DecisionTreeMiner(max_depth=3, random_state=SEED),
        ],
        num_classes=2, lr=0.005, max_iter=MAX_ITER, lossfn="CE",
        precompute_rules=True, force_precompute=True, base_breaks=2,
    )
    ensemble.fit(X_tr, y_tr, column_names=cols)
    return evaluate(ensemble.clf, X_te, y_te)


# ── Baselines ──
def get_baselines():
    from imodels import RuleFitClassifier, FIGSClassifier, GreedyRuleListClassifier
    return {
        "DecisionTree_d4": lambda: DecisionTreeClassifier(max_depth=4, random_state=SEED),
        "DecisionTree_d8": lambda: DecisionTreeClassifier(max_depth=8, random_state=SEED),
        "RuleFit": lambda: RuleFitClassifier(max_rules=30, tree_size=4, random_state=SEED),
        "FIGS": lambda: FIGSClassifier(max_rules=10),
        "GreedyRuleList": lambda: GreedyRuleListClassifier(max_depth=5),
        "GradientBoosting": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=SEED),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, max_depth=8, random_state=SEED),
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
    }


def run_baseline(name, clf_factory, X_tr, y_tr, X_te, y_te, cols):
    clf = clf_factory()
    try:
        if "RuleFit" in name:
            clf.fit(X_tr, y_tr, feature_names=cols)
        else:
            clf.fit(X_tr, y_tr)
        return evaluate(clf, X_te, y_te)
    except Exception as e:
        return {"accuracy": None, "f1": None, "roc_auc": None, "error": str(e)}


# ── Single dataset CV ──
def run_cv_for_dataset(dataset_name, n_folds, log):
    log.info("=" * 60)
    log.info("Dataset: %s (%d-fold CV)", dataset_name, n_folds)
    log.info("=" * 60)

    X, y, cols = load_dataset(dataset_name)
    log.info("  Samples: %d, Features: %d, Pos ratio: %.2f", len(y), len(cols), y.mean())

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    # Methods to evaluate
    dsgd_methods = {
        "DSGD_base": run_dsgd_base,
        "DSGD_augmented": run_dsgd_augmented,
        "DSGD_iterative": run_dsgd_iterative,
        "DSGD_ensemble": run_dsgd_ensemble,
    }
    baselines = get_baselines()

    all_methods = list(dsgd_methods.keys()) + list(baselines.keys())
    fold_results = {m: [] for m in all_methods}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        log.info("  Fold %d/%d", fold_idx + 1, n_folds)
        X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_tr_raw)
        X_tr = scaler.transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)

        # DSGD++ methods
        for name, fn in dsgd_methods.items():
            t0 = time.time()
            try:
                m = fn(X_tr, y_tr, X_te, y_te, cols)
                m["time"] = round(time.time() - t0, 1)
            except Exception as e:
                log.warning("    %s failed: %s", name, e)
                m = {"accuracy": None, "f1": None, "roc_auc": None, "error": str(e)}
            fold_results[name].append(m)
            if m.get("accuracy") is not None:
                log.info("    %s: acc=%.3f f1=%.3f (%.1fs)",
                         name, m["accuracy"], m["f1"], m.get("time", 0))

        # Baselines
        for name, factory in baselines.items():
            t0 = time.time()
            m = run_baseline(name, factory, X_tr, y_tr, X_te, y_te, cols)
            m["time"] = round(time.time() - t0, 1)
            fold_results[name].append(m)
            if m.get("accuracy") is not None:
                log.info("    %s: acc=%.3f f1=%.3f (%.1fs)",
                         name, m["accuracy"], m["f1"], m.get("time", 0))

    # Aggregate
    summary = {}
    for method, folds in fold_results.items():
        accs = [f["accuracy"] for f in folds if f.get("accuracy") is not None]
        f1s = [f["f1"] for f in folds if f.get("f1") is not None]
        aucs = [f["roc_auc"] for f in folds if f.get("roc_auc") is not None]
        if accs:
            summary[method] = {
                "accuracy_mean": round(np.mean(accs), 4),
                "accuracy_std": round(np.std(accs), 4),
                "f1_mean": round(np.mean(f1s), 4),
                "f1_std": round(np.std(f1s), 4),
                "auc_mean": round(np.mean(aucs), 4) if aucs else None,
                "auc_std": round(np.std(aucs), 4) if aucs else None,
                "n_folds": len(accs),
            }
        else:
            summary[method] = {"error": "all folds failed"}

    # Print summary
    log.info("")
    log.info("  %-25s  %s", "Method", "Accuracy (mean +/- std)")
    log.info("  " + "-" * 55)
    for method in all_methods:
        s = summary.get(method, {})
        if "accuracy_mean" in s:
            log.info("  %-25s  %.3f +/- %.3f",
                     method, s["accuracy_mean"], s["accuracy_std"])

    return fold_results, summary


# ── Wilcoxon Tests ──
def run_wilcoxon_tests(all_summaries, all_fold_results, log):
    """Compare DSGD++ best vs each baseline across datasets."""
    log.info("")
    log.info("=" * 60)
    log.info("Wilcoxon Signed-Rank Tests")
    log.info("=" * 60)

    datasets = list(all_fold_results.keys())
    methods = set()
    for ds_folds in all_fold_results.values():
        methods.update(ds_folds.keys())

    # Get mean accuracy per dataset for each method
    method_scores = {}
    for method in methods:
        scores = []
        for ds in datasets:
            s = all_summaries[ds].get(method, {})
            scores.append(s.get("accuracy_mean"))
        method_scores[method] = scores

    # Find DSGD++ best per dataset
    dsgd_methods = ["DSGD_base", "DSGD_augmented", "DSGD_iterative", "DSGD_ensemble"]
    dsgd_best = []
    dsgd_best_name = []
    for i, ds in enumerate(datasets):
        best_acc = 0
        best_name = ""
        for dm in dsgd_methods:
            acc = method_scores.get(dm, [None] * len(datasets))[i]
            if acc is not None and acc > best_acc:
                best_acc = acc
                best_name = dm
        dsgd_best.append(best_acc)
        dsgd_best_name.append(best_name)

    results = {}
    comparisons = [m for m in methods if m not in dsgd_methods]

    for baseline in sorted(comparisons):
        baseline_scores = method_scores[baseline]

        # Build paired arrays (skip None)
        pairs_dsgd = []
        pairs_base = []
        for d_score, b_score in zip(dsgd_best, baseline_scores):
            if d_score is not None and b_score is not None:
                pairs_dsgd.append(d_score)
                pairs_base.append(b_score)

        if len(pairs_dsgd) < 5:
            results[baseline] = {"error": f"only {len(pairs_dsgd)} paired observations"}
            continue

        diffs = [d - b for d, b in zip(pairs_dsgd, pairs_base)]
        mean_diff = np.mean(diffs)
        wins = sum(1 for d in diffs if d > 0.001)
        losses = sum(1 for d in diffs if d < -0.001)
        ties = len(diffs) - wins - losses

        try:
            stat, p_value = wilcoxon(pairs_dsgd, pairs_base, alternative="two-sided")
            results[baseline] = {
                "statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "significant": p_value < 0.05,
                "mean_diff": round(mean_diff, 4),
                "wins": wins, "losses": losses, "ties": ties,
                "n_datasets": len(pairs_dsgd),
            }
            sig = "*" if p_value < 0.05 else " "
            log.info("  DSGD++ vs %-20s  diff=%+.3f  W=%d L=%d T=%d  p=%.3f %s",
                     baseline, mean_diff, wins, losses, ties, p_value, sig)
        except Exception as e:
            results[baseline] = {"error": str(e)}
            log.info("  DSGD++ vs %-20s  %s", baseline, e)

    # Also test DSGD_iterative vs DSGD_base (does refinement help?)
    base_scores = method_scores.get("DSGD_base", [])
    iter_scores = method_scores.get("DSGD_iterative", [])
    pairs = [(b, i) for b, i in zip(base_scores, iter_scores) if b is not None and i is not None]
    if len(pairs) >= 5:
        b_arr, i_arr = zip(*pairs)
        try:
            stat, p_value = wilcoxon(i_arr, b_arr, alternative="greater")
            results["_ablation_iterative_vs_base"] = {
                "statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "significant": p_value < 0.05,
                "mean_diff": round(np.mean([i - b for b, i in pairs]), 4),
            }
            log.info("")
            log.info("  Ablation: iterative vs base  diff=%+.3f  p=%.3f %s",
                     np.mean([i - b for b, i in pairs]), p_value,
                     "*" if p_value < 0.05 else "")
        except Exception as e:
            results["_ablation_iterative_vs_base"] = {"error": str(e)}

    return results


# ── Ablation summary ──
def build_ablation_summary(all_summaries, log):
    """Show progressive improvement: base → +mining → +iterative → +ensemble."""
    log.info("")
    log.info("=" * 60)
    log.info("Ablation Study")
    log.info("=" * 60)

    stages = ["DSGD_base", "DSGD_augmented", "DSGD_iterative", "DSGD_ensemble"]
    stage_labels = ["Base (thresholds)", "+ DT mining", "+ Iterative refine", "+ Multi-source"]

    ablation = {}
    log.info("  %-25s  %s", "Dataset", "  ".join(f"{l:<18}" for l in stage_labels))
    log.info("  " + "-" * 100)

    for ds, summary in all_summaries.items():
        row = {}
        parts = []
        for stage, label in zip(stages, stage_labels):
            s = summary.get(stage, {})
            acc = s.get("accuracy_mean")
            std = s.get("accuracy_std")
            if acc is not None:
                row[stage] = {"mean": acc, "std": std}
                parts.append(f"{acc:.3f}+/-{std:.3f}")
            else:
                row[stage] = None
                parts.append("  N/A           ")
        ablation[ds] = row
        log.info("  %-25s  %s", ds, "  ".join(parts))

    return ablation


# ── Main ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=None,
                        help="Dataset names (default: all available)")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    import torch
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    log = setup_logging()

    # Discover datasets
    all_datasets = [
        "heart-disease", "breast-cancer-wisconsin", "ionosphere",
        "pima-diabetes", "qsar-biodeg", "phoneme",
        "sonar", "banknote-authentication", "blood-transfusion", "german-credit",
    ]
    if args.datasets:
        datasets = args.datasets
    else:
        # Use only datasets that exist
        datasets = [d for d in all_datasets
                    if os.path.exists(os.path.join("src", "datasets", f"{d}.csv"))]

    log.info("Datasets: %s", datasets)
    log.info("Folds: %d", args.folds)

    # Output dir
    out_dir = os.path.join("experiment_results", "cv_results")
    os.makedirs(out_dir, exist_ok=True)

    all_fold_results = {}
    all_summaries = {}

    for ds in datasets:
        # Check cache
        ds_dir = os.path.join(out_dir, ds)
        cache_file = os.path.join(ds_dir, "fold_results.json")
        if os.path.exists(cache_file):
            log.info("  [cache hit] %s", ds)
            with open(cache_file) as f:
                cached = json.load(f)
            all_fold_results[ds] = cached["fold_results"]
            all_summaries[ds] = cached["summary"]
            continue

        fold_results, summary = run_cv_for_dataset(ds, args.folds, log)
        all_fold_results[ds] = fold_results
        all_summaries[ds] = summary

        # Save per-dataset
        os.makedirs(ds_dir, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump({"fold_results": fold_results, "summary": summary}, f, indent=2, default=str)

    # Cross-dataset analysis
    wilcoxon_results = run_wilcoxon_tests(all_summaries, all_fold_results, log)
    ablation = build_ablation_summary(all_summaries, log)

    # Save global results
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    with open(os.path.join(out_dir, "wilcoxon_tests.json"), "w") as f:
        json.dump(wilcoxon_results, f, indent=2, default=str)
    with open(os.path.join(out_dir, "ablation.json"), "w") as f:
        json.dump(ablation, f, indent=2, default=str)

    # Final summary table
    log.info("")
    log.info("=" * 70)
    log.info("FINAL CROSS-DATASET SUMMARY (%d-fold CV)", args.folds)
    log.info("=" * 70)
    log.info("%-25s  %-20s  %-20s  %-20s", "Dataset", "DSGD++ best", "RuleFit", "RandomForest")
    log.info("-" * 90)
    for ds in datasets:
        s = all_summaries[ds]
        # DSGD best
        dsgd_best = 0
        dsgd_name = ""
        for dm in ["DSGD_base", "DSGD_augmented", "DSGD_iterative", "DSGD_ensemble"]:
            acc = s.get(dm, {}).get("accuracy_mean", 0)
            if acc and acc > dsgd_best:
                dsgd_best = acc
                dsgd_name = dm
        rf_acc = s.get("RuleFit", {}).get("accuracy_mean", 0)
        rfo_acc = s.get("RandomForest", {}).get("accuracy_mean", 0)
        dsgd_std = s.get(dsgd_name, {}).get("accuracy_std", 0)
        rf_std = s.get("RuleFit", {}).get("accuracy_std", 0)
        rfo_std = s.get("RandomForest", {}).get("accuracy_std", 0)
        log.info("%-25s  %.3f+/-%.3f      %.3f+/-%.3f      %.3f+/-%.3f",
                 ds, dsgd_best, dsgd_std, rf_acc, rf_std, rfo_acc, rfo_std)

    log.info("")
    log.info("Results saved to %s", out_dir)


if __name__ == "__main__":
    main()
