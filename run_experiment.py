"""
Generic experiment runner for the rule induction paper.

Usage:
    python run_experiment.py breast-cancer-wisconsin
    python run_experiment.py heart-disease
    python run_experiment.py --max-iter 200 breast-cancer-wisconsin

Monitor: tail -f experiment_results/<dataset>/experiment.log
"""
import argparse
import json
import logging
import logging.handlers
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.DSClassifierMultiQ import DSClassifierMultiQ
from src.rule_adapter import build_feature_name_map
from src.rule_miners import SkopeRulesMiner, RipperMiner, DecisionTreeMiner
from src.uncertainty_loop import UncertaintyGuidedRefiner
from src.rule_ensemble import MultiSourceEnsemble
from src.rule_pruning import ConfidenceBasedPruner

SEED = 509


def setup_dirs(dataset_name):
    base = os.path.join("experiment_results", dataset_name)
    dirs = {
        "results": base,
        "cache": os.path.join(base, "_cache"),
        "rules": os.path.join(base, "rules"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def setup_logging(dirs):
    log = logging.getLogger("experiment")
    log.handlers.clear()
    log.setLevel(logging.DEBUG)

    class FlushFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    fh = FlushFileHandler(os.path.join(dirs["results"], "experiment.log"), mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))

    log.addHandler(fh)
    log.addHandler(ch)
    sys.stdout.reconfigure(line_buffering=True)
    return log


# ---- Cache ----
def cache_path(dirs, name):
    return os.path.join(dirs["cache"], f"{name}.json")

def load_cache(dirs, log, name):
    p = cache_path(dirs, name)
    if os.path.exists(p):
        with open(p) as f:
            log.info("  [cache hit] %s", name)
            return json.load(f)
    return None

def save_cache(dirs, name, data):
    with open(cache_path(dirs, name), "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---- Data ----
def load_dataset(dataset_name):
    path = os.path.join("src", "datasets", f"{dataset_name}.csv")
    df = pd.read_csv(path)
    y = df["labels"].values
    X = df.drop(columns=["labels"]).values
    col_names = [c for c in df.columns if c != "labels"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test, col_names


# ---- Helpers ----
def evaluate(clf, X_test, y_test):
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba[:, 1]), 4),
    }

def save_result(dirs, name, result):
    path = os.path.join(dirs["results"], f"{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)

def save_rules_and_scores(dirs, label, clf):
    scores = clf.model.get_rule_uncertainty_scores()
    rules_data = [{
        "rule_idx": s["rule_idx"],
        "caption": s["caption"],
        "uncertainty_mass": round(s["uncertainty_mass"], 6),
        "max_class_mass": round(s["max_class_mass"], 6),
        "confidence_score": round(s["confidence_score"], 6),
        "masses": [round(float(m), 6) for m in s["masses"]],
    } for s in scores]
    path = os.path.join(dirs["rules"], f"{label}_rules.json")
    with open(path, "w") as f:
        json.dump(rules_data, f, indent=2)


def make_clf(max_iter=500, lr=0.005):
    return DSClassifierMultiQ(
        num_classes=2, lr=lr, max_iter=max_iter, lossfn="CE",
        precompute_rules=True, force_precompute=True,
    )


# ---- E1: Rule source comparison ----
def run_e1(dirs, log, X_tr, X_te, y_tr, y_te, cols, max_iter):
    log.info("=" * 60)
    log.info("E1: Rule Source Comparison")
    log.info("=" * 60)
    fm = build_feature_name_map(cols)
    results = {}

    configs = [
        ("single_feature", []),
        ("skope_rules", [SkopeRulesMiner(n_estimators=50, precision_min=0.3, recall_min=0.01, max_depth=3)]),
        ("ripper", [RipperMiner()]),
        ("decision_tree", [DecisionTreeMiner(max_depth=4, random_state=SEED)]),
    ]

    for label, miners in configs:
        key = f"E1_{label}"
        cached = load_cache(dirs, log, key)
        if cached is not None:
            results[label] = cached
            continue

        t0 = time.time()
        clf = make_clf(max_iter)
        clf.model.generate_statistic_single_rules(X_tr, breaks=2, column_names=cols)
        base = clf.model.get_rules_size()
        mined = 0
        for miner in miners:
            try:
                miner.fit(X_tr, y_tr, feature_names=cols)
                for r in miner.extract_rules(fm):
                    clf.model.add_rule(r, method="random")
                    mined += 1
            except Exception as e:
                log.warning("  %s: %s failed: %s", key, type(miner).__name__, e)

        log.info("  %s: training %d rules (%d base + %d mined)...", key, base + mined, base, mined)
        clf.fit(X_tr, y_tr)
        dt = time.time() - t0

        m = evaluate(clf, X_te, y_te)
        m["n_rules"] = clf.model.get_rules_size()
        m["training_time"] = round(dt, 2)
        log.info("  %s: acc=%.3f f1=%.3f auc=%.3f rules=%d time=%.1fs",
                 key, m["accuracy"], m["f1"], m["roc_auc"], m["n_rules"], dt)

        save_rules_and_scores(dirs, key, clf)
        save_cache(dirs, key, m)
        results[label] = m

    save_result(dirs, "E1_rule_source_comparison", results)
    return results


# ---- E2: Iterative refinement ----
def run_e2(dirs, log, X_tr, X_te, y_tr, y_te, cols, max_iter):
    log.info("=" * 60)
    log.info("E2: Iterative Uncertainty-Guided Refinement")
    log.info("=" * 60)
    results = {}

    base_rules = len(cols) * 3  # approximate: features * (breaks+1)
    total_cap = base_rules * 3  # allow up to 3x base

    for miner_name, miner in [
        ("SkopeRules", SkopeRulesMiner(n_estimators=50, precision_min=0.3, recall_min=0.01, max_rules=5)),
        ("DecisionTree", DecisionTreeMiner(max_depth=3, random_state=SEED, max_rules=5)),
    ]:
        key = f"E2_{miner_name}"
        cached = load_cache(dirs, log, key)
        if cached is not None:
            results[miner_name] = cached
            continue

        log.info("  Refiner with %s (cap=%d total rules)...", miner_name, total_cap)
        t0 = time.time()
        clf = make_clf(max_iter)
        refiner = UncertaintyGuidedRefiner(
            clf, miner,
            uncertainty_threshold=0.3,
            max_iterations=3,
            min_covered_samples=10,
            max_rules_per_weak=3,
            max_total_rules=total_cap,
            max_weak_to_refine=10,
        )
        result = refiner.refine(X_tr, y_tr, column_names=cols, single_rules_breaks=2)
        dt = time.time() - t0

        m = evaluate(result["clf"], X_te, y_te)
        m["initial_rules"] = result["initial_rules"]
        m["final_rules"] = result["final_rules"]
        m["iterations"] = len(result["history"])
        m["history"] = result["history"]
        m["training_time"] = round(dt, 2)

        log.info("  %s: acc=%.3f f1=%.3f rules=%d->%d (%d iters) time=%.1fs",
                 miner_name, m["accuracy"], m["f1"],
                 m["initial_rules"], m["final_rules"], m["iterations"], dt)

        save_rules_and_scores(dirs, key, result["clf"])
        save_cache(dirs, key, m)
        results[miner_name] = m

    save_result(dirs, "E2_iterative_refinement", results)
    return results


# ---- E3: Multi-source ensemble ----
def run_e3(dirs, log, X_tr, X_te, y_tr, y_te, cols, max_iter):
    log.info("=" * 60)
    log.info("E3: Multi-Source Rule Ensemble")
    log.info("=" * 60)
    results = {}

    # Single-source baseline
    key = "E3_single_source"
    cached = load_cache(dirs, log, key)
    if cached is not None:
        results["single_source"] = cached
    else:
        t0 = time.time()
        clf = make_clf(max_iter)
        clf.fit(X_tr, y_tr, add_single_rules=True, single_rules_breaks=2, column_names=cols)
        dt = time.time() - t0
        m = evaluate(clf, X_te, y_te)
        m["n_rules"] = clf.model.get_rules_size()
        m["training_time"] = round(dt, 2)
        log.info("  single_source: acc=%.3f f1=%.3f rules=%d time=%.1fs",
                 m["accuracy"], m["f1"], m["n_rules"], dt)
        save_rules_and_scores(dirs, key, clf)
        save_cache(dirs, key, m)
        results["single_source"] = m

    # Multi-source
    key = "E3_multi_source"
    cached = load_cache(dirs, log, key)
    if cached is not None:
        results["multi_source"] = cached
    else:
        log.info("  Multi-source ensemble...")
        t0 = time.time()
        ensemble = MultiSourceEnsemble(
            miners=[
                SkopeRulesMiner(n_estimators=50, precision_min=0.3, recall_min=0.01, max_rules=10),
                RipperMiner(max_rules=10),
                DecisionTreeMiner(max_depth=3, random_state=SEED),
            ],
            num_classes=2, lr=0.005, max_iter=max_iter, lossfn="CE",
            precompute_rules=True, force_precompute=True,
            base_breaks=2,
        )
        ensemble.fit(X_tr, y_tr, column_names=cols)
        dt = time.time() - t0
        m = evaluate(ensemble.clf, X_te, y_te)
        m["n_rules"] = ensemble.clf.model.get_rules_size()
        m["rule_sources"] = ensemble.get_rule_sources()
        m["training_time"] = round(dt, 2)
        log.info("  multi_source: acc=%.3f f1=%.3f rules=%d sources=%s time=%.1fs",
                 m["accuracy"], m["f1"], m["n_rules"], m["rule_sources"], dt)
        save_rules_and_scores(dirs, key, ensemble.clf)
        save_cache(dirs, key, m)
        results["multi_source"] = m

    save_result(dirs, "E3_ensemble", results)
    return results


# ---- E4: Pruning Pareto ----
def run_e4(dirs, log, X_tr, X_te, y_tr, y_te, cols, max_iter):
    log.info("=" * 60)
    log.info("E4: Pruning Pareto Frontier")
    log.info("=" * 60)

    cached = load_cache(dirs, log, "E4_pareto")
    if cached is not None:
        save_result(dirs, "E4_pruning_pareto", cached)
        return cached

    log.info("  Training base model (ensemble)...")
    t0 = time.time()
    ensemble = MultiSourceEnsemble(
        miners=[
            SkopeRulesMiner(n_estimators=50, precision_min=0.3, recall_min=0.01),
            DecisionTreeMiner(max_depth=4, random_state=SEED),
        ],
        num_classes=2, lr=0.005, max_iter=max_iter, lossfn="CE",
        precompute_rules=True, force_precompute=True, base_breaks=2,
    )
    ensemble.fit(X_tr, y_tr, column_names=cols)
    log.info("  Base model: %d rules (%.1fs)", ensemble.clf.model.get_rules_size(), time.time() - t0)

    results = {}
    thresholds = np.linspace(0, 0.45, 15)
    for strategy in ["confidence", "random"]:
        log.info("  Pruning: %s...", strategy)
        pruner = ConfidenceBasedPruner(strategy=strategy, random_state=SEED)
        pareto = pruner.pareto_frontier(
            ensemble.clf, X_tr, y_tr, X_te, y_te, thresholds=thresholds)
        results[strategy] = pareto
        for p in pareto[::4]:
            log.info("    t=%.2f: %d rules, acc=%.3f", p["threshold"], p["n_active_rules"], p["accuracy"])

    save_cache(dirs, "E4_pareto", results)
    save_result(dirs, "E4_pruning_pareto", results)
    return results


# ---- E5: Standalone baselines ----
def run_e5(dirs, log, X_tr, X_te, y_tr, y_te, cols):
    log.info("=" * 60)
    log.info("E5: Interpretable Baselines")
    log.info("=" * 60)

    from imodels import SkopeRulesClassifier, RuleFitClassifier, GreedyRuleListClassifier, FIGSClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import wittgenstein as lw

    results = {}

    baselines = [
        ("DecisionTree_d4", DecisionTreeClassifier(max_depth=4, random_state=SEED)),
        ("DecisionTree_d8", DecisionTreeClassifier(max_depth=8, random_state=SEED)),
        ("SkopeRules", SkopeRulesClassifier(n_estimators=50, precision_min=0.3, recall_min=0.01, max_depth=3)),
        ("RuleFit", RuleFitClassifier(max_rules=30, tree_size=4, random_state=SEED)),
        ("GreedyRuleList", GreedyRuleListClassifier(max_depth=5)),
        ("FIGS", FIGSClassifier(max_rules=10)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=SEED)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=SEED)),
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=SEED)),
    ]

    for name, clf in baselines:
        key = f"E5_{name}"
        cached = load_cache(dirs, log, key)
        if cached is not None:
            results[name] = cached
            continue
        log.info("  %s...", name)
        t0 = time.time()
        try:
            if "Skope" in name or "RuleFit" in name:
                clf.fit(X_tr, y_tr, feature_names=cols)
            else:
                clf.fit(X_tr, y_tr)
            dt = time.time() - t0
            preds = clf.predict(X_te)
            proba = clf.predict_proba(X_te) if hasattr(clf, "predict_proba") else None
            m = {
                "accuracy": round(accuracy_score(y_te, preds), 4),
                "f1": round(f1_score(y_te, preds, zero_division=0), 4),
                "precision": round(precision_score(y_te, preds, zero_division=0), 4),
                "recall": round(recall_score(y_te, preds, zero_division=0), 4),
                "roc_auc": round(roc_auc_score(y_te, proba[:, 1]), 4) if proba is not None else None,
                "training_time": round(dt, 2),
            }
            log.info("  %s: acc=%.3f f1=%.3f time=%.1fs", name, m["accuracy"], m["f1"], dt)
        except Exception as e:
            log.warning("  %s failed: %s", name, e)
            m = {"error": str(e)}
        save_cache(dirs, key, m)
        results[name] = m

    # RIPPER (needs DataFrame)
    key = "E5_RIPPER"
    cached = load_cache(dirs, log, key)
    if cached is not None:
        results["RIPPER"] = cached
    else:
        log.info("  RIPPER...")
        t0 = time.time()
        try:
            df_tr = pd.DataFrame(X_tr, columns=cols)
            df_tr["_target"] = y_tr
            df_te = pd.DataFrame(X_te, columns=cols)
            ripper = lw.RIPPER(prune_size=0.33, k=2)
            unique, counts = np.unique(y_tr, return_counts=True)
            ripper.fit(df_tr, class_feat="_target", pos_class=unique[np.argmin(counts)])
            preds = ripper.predict(df_te)
            m = {
                "accuracy": round(accuracy_score(y_te, preds), 4),
                "f1": round(f1_score(y_te, preds, zero_division=0), 4),
                "precision": round(precision_score(y_te, preds, zero_division=0), 4),
                "recall": round(recall_score(y_te, preds, zero_division=0), 4),
                "roc_auc": None,
                "training_time": round(time.time() - t0, 2),
            }
            log.info("  RIPPER: acc=%.3f f1=%.3f time=%.1fs", m["accuracy"], m["f1"], m["training_time"])
        except Exception as e:
            log.warning("  RIPPER failed: %s", e)
            m = {"error": str(e)}
        save_cache(dirs, key, m)
        results["RIPPER"] = m

    save_result(dirs, "E5_baselines", results)
    return results


# ---- Summary ----
def print_summary(log, all_results):
    log.info("")
    log.info("=" * 72)
    log.info("SUMMARY")
    log.info("=" * 72)
    log.info("%-35s %6s %6s %6s %6s %6s", "Method", "Acc", "F1", "AUC", "Rules", "Time")
    log.info("-" * 72)
    for method, m in all_results.get("E1", {}).items():
        log.info("  E1 %-30s %6.3f %6.3f %6.3f %6d %5.1fs",
                 method, m["accuracy"], m["f1"], m["roc_auc"], m["n_rules"], m["training_time"])
    for method, m in all_results.get("E2", {}).items():
        log.info("  E2 %-30s %6.3f %6.3f %6.3f %6d %5.1fs",
                 method, m["accuracy"], m["f1"], m["roc_auc"], m["final_rules"], m["training_time"])
    for method, m in all_results.get("E3", {}).items():
        log.info("  E3 %-30s %6.3f %6.3f %6.3f %6d %5.1fs",
                 method, m["accuracy"], m["f1"], m["roc_auc"], m["n_rules"], m["training_time"])
    for method, m in all_results.get("E5", {}).items():
        if "error" in m:
            log.info("  E5 %-30s FAILED", method)
        else:
            auc = f'{m["roc_auc"]:.3f}' if m.get("roc_auc") else "  N/A"
            log.info("  E5 %-30s %6.3f %6.3f %6s %6s %5.1fs",
                     method, m["accuracy"], m["f1"], auc, "-", m["training_time"])
    log.info("=" * 72)


# ---- Main ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rule induction experiments on a dataset")
    parser.add_argument("dataset", help="Dataset name (without .csv), e.g. breast-cancer-wisconsin")
    parser.add_argument("--max-iter", type=int, default=500, help="Max DSGD++ epochs")
    args = parser.parse_args()

    import torch
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dirs = setup_dirs(args.dataset)
    log = setup_logging(dirs)

    log.info("Dataset: %s", args.dataset)
    X_tr, X_te, y_tr, y_te, cols = load_dataset(args.dataset)
    log.info("  Train: %s, Test: %s, Features: %d", X_tr.shape, X_te.shape, len(cols))

    R = {}
    R["E1"] = run_e1(dirs, log, X_tr, X_te, y_tr, y_te, cols, args.max_iter)
    R["E2"] = run_e2(dirs, log, X_tr, X_te, y_tr, y_te, cols, args.max_iter)
    R["E3"] = run_e3(dirs, log, X_tr, X_te, y_tr, y_te, cols, args.max_iter)
    R["E4"] = run_e4(dirs, log, X_tr, X_te, y_tr, y_te, cols, args.max_iter)
    R["E5"] = run_e5(dirs, log, X_tr, X_te, y_tr, y_te, cols)

    print_summary(log, R)
