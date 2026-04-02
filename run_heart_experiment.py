"""
Run the full rule induction experiment suite on the Heart Disease dataset.

E1: Rule source comparison (single-feature vs RIPPER vs SkopeRules vs DecisionTree)
E2: Iterative uncertainty-guided refinement
E3: Multi-source rule ensemble
E4: Pruning Pareto frontier

Usage:
    python run_heart_experiment.py              # full run
    python run_heart_experiment.py --quick      # quick subset estimate first

Monitor progress:
    tail -f experiment_results/heart-disease/experiment.log
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
RESULTS_DIR = "experiment_results/heart-disease"
CACHE_DIR = os.path.join(RESULTS_DIR, "_cache")
LOG_FILE = os.path.join(RESULTS_DIR, "experiment.log")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ---- Logging ----
log = logging.getLogger("experiment")
log.setLevel(logging.DEBUG)
# File handler (detailed, auto-flush)
class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

fh = FlushFileHandler(LOG_FILE, mode="a")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
# Console handler (concise, auto-flush)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
log.addHandler(fh)
log.addHandler(ch)
# Also force stdout unbuffered
sys.stdout.reconfigure(line_buffering=True)


# ---- Cache helpers ----
def cache_path(name):
    return os.path.join(CACHE_DIR, f"{name}.json")


def load_cache(name):
    p = cache_path(name)
    if os.path.exists(p):
        with open(p) as f:
            log.info("  [cache hit] Loading %s from cache", name)
            return json.load(f)
    return None


def save_cache(name, data):
    with open(cache_path(name), "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---- Data ----
def load_and_split():
    df = pd.read_csv("src/datasets/heart-disease.csv")
    y = df["labels"].values
    X = df.drop(columns=["labels"]).values
    col_names = [c for c in df.columns if c != "labels"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, col_names


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


def save_result(name, result):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info("  Saved: %s", path)


# ---- Train a single DSGD++ variant and return metrics ----
def train_and_eval(label, X_train, y_train, X_test, y_test, col_names,
                   extra_miners=None, max_iter=500, lr=0.005, breaks=2):
    """Train DSGD++ with optional mined rules. Returns metrics dict."""
    cache = load_cache(label)
    if cache is not None:
        return cache

    fm = build_feature_name_map(col_names)
    t0 = time.time()

    clf = DSClassifierMultiQ(num_classes=2, lr=lr, max_iter=max_iter, lossfn="CE",
                              precompute_rules=True, force_precompute=True)
    clf.model.generate_statistic_single_rules(X_train, breaks=breaks, column_names=col_names)
    base_rules = clf.model.get_rules_size()
    log.debug("  %s: generated %d base rules", label, base_rules)

    mined_count = 0
    if extra_miners:
        for miner in extra_miners:
            mname = type(miner).__name__
            log.debug("  %s: mining with %s...", label, mname)
            try:
                miner.fit(X_train, y_train, feature_names=col_names)
                rules = miner.extract_rules(fm)
                for r in rules:
                    clf.model.add_rule(r, method="random")
                mined_count += len(rules)
                log.debug("  %s: %s produced %d rules", label, mname, len(rules))
            except Exception as e:
                log.warning("  %s: %s failed: %s", label, mname, e)

    total_rules = clf.model.get_rules_size()
    log.info("  %s: training DSGD++ with %d rules (%d base + %d mined)...",
             label, total_rules, base_rules, mined_count)

    clf.fit(X_train, y_train)
    dt = time.time() - t0

    metrics = evaluate(clf, X_test, y_test)
    metrics["n_rules"] = total_rules
    metrics["training_time"] = round(dt, 2)
    log.info("  %s: acc=%.3f f1=%.3f auc=%.3f rules=%d time=%.1fs",
             label, metrics["accuracy"], metrics["f1"], metrics["roc_auc"],
             metrics["n_rules"], dt)

    save_cache(label, metrics)
    return metrics


# ============================================================
# E1: Rule source comparison
# ============================================================
def run_e1(X_train, X_test, y_train, y_test, col_names, max_iter=500):
    log.info("=" * 60)
    log.info("E1: Rule Source Comparison")
    log.info("=" * 60)

    results = {}

    results["single_feature"] = train_and_eval(
        "E1_single_feature", X_train, y_train, X_test, y_test, col_names,
        max_iter=max_iter)

    results["skope_rules"] = train_and_eval(
        "E1_skope_rules", X_train, y_train, X_test, y_test, col_names,
        extra_miners=[SkopeRulesMiner(n_estimators=50, precision_min=0.3, recall_min=0.01, max_depth=3)],
        max_iter=max_iter)

    results["ripper"] = train_and_eval(
        "E1_ripper", X_train, y_train, X_test, y_test, col_names,
        extra_miners=[RipperMiner()],
        max_iter=max_iter)

    results["decision_tree"] = train_and_eval(
        "E1_decision_tree", X_train, y_train, X_test, y_test, col_names,
        extra_miners=[DecisionTreeMiner(max_depth=4, random_state=SEED)],
        max_iter=max_iter)

    save_result("E1_rule_source_comparison", results)
    return results


# ============================================================
# E2: Iterative refinement
# ============================================================
def run_e2(X_train, X_test, y_train, y_test, col_names, max_iter=500):
    log.info("=" * 60)
    log.info("E2: Iterative Uncertainty-Guided Refinement")
    log.info("=" * 60)

    results = {}

    for miner_name, miner in [
        ("SkopeRules", SkopeRulesMiner(n_estimators=50, precision_min=0.3, recall_min=0.01, max_rules=5)),
        ("DecisionTree", DecisionTreeMiner(max_depth=3, random_state=SEED, max_rules=5)),
    ]:
        cache_key = f"E2_{miner_name}"
        cached = load_cache(cache_key)
        if cached is not None:
            results[miner_name] = cached
            continue

        log.info("  Refiner with %s...", miner_name)
        t0 = time.time()
        clf = DSClassifierMultiQ(num_classes=2, lr=0.005, max_iter=max_iter, lossfn="CE",
                                  precompute_rules=True, force_precompute=True)
        refiner = UncertaintyGuidedRefiner(
            clf, miner,
            uncertainty_threshold=0.3,
            max_iterations=3,
            min_covered_samples=10,
            max_rules_per_weak=3,
        )
        result = refiner.refine(X_train, y_train, column_names=col_names, single_rules_breaks=2)
        dt = time.time() - t0

        metrics = evaluate(result["clf"], X_test, y_test)
        metrics["initial_rules"] = result["initial_rules"]
        metrics["final_rules"] = result["final_rules"]
        metrics["iterations"] = len(result["history"])
        metrics["history"] = result["history"]
        metrics["training_time"] = round(dt, 2)

        log.info("  %s: acc=%.3f f1=%.3f rules=%d->%d (%d iters) time=%.1fs",
                 miner_name, metrics["accuracy"], metrics["f1"],
                 metrics["initial_rules"], metrics["final_rules"],
                 metrics["iterations"], dt)

        results[miner_name] = metrics
        save_cache(cache_key, metrics)

    save_result("E2_iterative_refinement", results)
    return results


# ============================================================
# E3: Multi-source ensemble
# ============================================================
def run_e3(X_train, X_test, y_train, y_test, col_names, max_iter=500):
    log.info("=" * 60)
    log.info("E3: Multi-Source Rule Ensemble")
    log.info("=" * 60)

    results = {}

    # Single-source baseline
    results["single_source"] = train_and_eval(
        "E3_single_source", X_train, y_train, X_test, y_test, col_names,
        max_iter=max_iter)

    # Multi-source
    cache_key = "E3_multi_source"
    cached = load_cache(cache_key)
    if cached is not None:
        results["multi_source"] = cached
    else:
        log.info("  Multi-source ensemble (SkopeRules + RIPPER + DecisionTree)...")
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
        ensemble.fit(X_train, y_train, column_names=col_names)
        dt = time.time() - t0
        metrics = evaluate(ensemble.clf, X_test, y_test)
        metrics["n_rules"] = ensemble.clf.model.get_rules_size()
        metrics["rule_sources"] = ensemble.get_rule_sources()
        metrics["training_time"] = round(dt, 2)

        log.info("  multi_source: acc=%.3f f1=%.3f rules=%d sources=%s time=%.1fs",
                 metrics["accuracy"], metrics["f1"], metrics["n_rules"],
                 metrics["rule_sources"], dt)

        results["multi_source"] = metrics
        save_cache(cache_key, metrics)

    save_result("E3_ensemble", results)
    return results


# ============================================================
# E4: Pruning Pareto frontier
# ============================================================
def run_e4(X_train, X_test, y_train, y_test, col_names, max_iter=500):
    log.info("=" * 60)
    log.info("E4: Pruning Pareto Frontier")
    log.info("=" * 60)

    cached = load_cache("E4_pareto")
    if cached is not None:
        save_result("E4_pruning_pareto", cached)
        return cached

    log.info("  Training base model (ensemble with many rules)...")
    t0 = time.time()
    ensemble = MultiSourceEnsemble(
        miners=[
            SkopeRulesMiner(n_estimators=50, precision_min=0.3, recall_min=0.01),
            DecisionTreeMiner(max_depth=4, random_state=SEED),
        ],
        num_classes=2, lr=0.005, max_iter=max_iter, lossfn="CE",
        precompute_rules=True, force_precompute=True,
        base_breaks=2,
    )
    ensemble.fit(X_train, y_train, column_names=col_names)
    log.info("  Base model: %d rules (%.1fs)", ensemble.clf.model.get_rules_size(), time.time() - t0)

    results = {}
    thresholds = np.linspace(0, 0.45, 15)

    for strategy in ["confidence", "random"]:
        log.info("  Pruning with %s strategy...", strategy)
        pruner = ConfidenceBasedPruner(strategy=strategy, random_state=SEED)
        pareto = pruner.pareto_frontier(
            ensemble.clf, X_train, y_train, X_test, y_test,
            thresholds=thresholds,
        )
        results[strategy] = pareto
        for p in pareto[::4]:
            log.info("    t=%.2f: %d rules, acc=%.3f, f1=%.3f",
                     p["threshold"], p["n_active_rules"], p["accuracy"], p["f1"])

    save_cache("E4_pareto", results)
    save_result("E4_pruning_pareto", results)
    return results


# ============================================================
# Quick timing estimate
# ============================================================
def run_quick_estimate(X_train, X_test, y_train, y_test, col_names):
    """Run a fast mini-experiment to estimate total runtime."""
    log.info("=" * 60)
    log.info("QUICK TIMING ESTIMATE (50 samples, 50 epochs)")
    log.info("=" * 60)

    n = min(50, len(X_train))
    X_sub, y_sub = X_train[:n], y_train[:n]
    quick_iter = 50

    timings = {}

    # Single-feature baseline
    t0 = time.time()
    clf = DSClassifierMultiQ(num_classes=2, lr=0.01, max_iter=quick_iter, lossfn="CE",
                              precompute_rules=True, force_precompute=True)
    clf.fit(X_sub, y_sub, add_single_rules=True, single_rules_breaks=2, column_names=col_names)
    timings["single_feature_50ep"] = time.time() - t0
    log.info("  Single-feature (50 samples, 50 epochs): %.2fs", timings["single_feature_50ep"])

    # SkopeRules mining
    t0 = time.time()
    miner = SkopeRulesMiner(n_estimators=30, precision_min=0.3, recall_min=0.01)
    miner.fit(X_sub, y_sub, feature_names=col_names)
    fm = build_feature_name_map(col_names)
    rules = miner.extract_rules(fm)
    timings["skope_mining"] = time.time() - t0
    log.info("  SkopeRules mining: %.2fs (%d rules)", timings["skope_mining"], len(rules))

    # RIPPER mining
    t0 = time.time()
    miner = RipperMiner()
    miner.fit(X_sub, y_sub, feature_names=col_names)
    rules = miner.extract_rules(fm)
    timings["ripper_mining"] = time.time() - t0
    log.info("  RIPPER mining: %.2fs (%d rules)", timings["ripper_mining"], len(rules))

    # Full-data training estimate
    t0 = time.time()
    clf = DSClassifierMultiQ(num_classes=2, lr=0.005, max_iter=quick_iter, lossfn="CE",
                              precompute_rules=True, force_precompute=True)
    clf.fit(X_train, y_train, add_single_rules=True, single_rules_breaks=2, column_names=col_names)
    timings["full_data_50ep"] = time.time() - t0
    log.info("  Full data (%d samples, 50 epochs): %.2fs", len(X_train), timings["full_data_50ep"])

    # Estimate total
    scale_factor = 500 / 50  # 500 epochs vs 50
    est_per_train = timings["full_data_50ep"] * scale_factor
    # E1: 4 trains, E2: ~4 trains (2 miners x ~2 iters), E3: 2 trains, E4: 1 train + 30 evals
    est_total = est_per_train * (4 + 4 + 2 + 1) + 30 * timings["full_data_50ep"]
    log.info("")
    log.info("  Estimated time per 500-epoch train: %.1fs", est_per_train)
    log.info("  Estimated total experiment time: %.0fs (%.1f min)", est_total, est_total / 60)

    return timings


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run quick timing estimate first")
    parser.add_argument("--max-iter", type=int, default=500, help="Max DSGD++ epochs")
    args = parser.parse_args()

    import torch
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    log.info("Loading Heart Disease dataset...")
    X_train, X_test, y_train, y_test, col_names = load_and_split()
    log.info("  Train: %s, Test: %s", X_train.shape, X_test.shape)
    log.info("  Features: %s", col_names)

    if args.quick:
        run_quick_estimate(X_train, X_test, y_train, y_test, col_names)
        log.info("\nQuick estimate done. Run without --quick for the full experiment.")
        sys.exit(0)

    all_results = {}
    all_results["E1"] = run_e1(X_train, X_test, y_train, y_test, col_names, max_iter=args.max_iter)
    all_results["E2"] = run_e2(X_train, X_test, y_train, y_test, col_names, max_iter=args.max_iter)
    all_results["E3"] = run_e3(X_train, X_test, y_train, y_test, col_names, max_iter=args.max_iter)
    all_results["E4"] = run_e4(X_train, X_test, y_train, y_test, col_names, max_iter=args.max_iter)

    # Summary table
    log.info("")
    log.info("=" * 72)
    log.info("SUMMARY")
    log.info("=" * 72)
    log.info("%-35s %6s %6s %6s %6s %6s", "Method", "Acc", "F1", "AUC", "Rules", "Time")
    log.info("-" * 72)
    for method, m in all_results["E1"].items():
        log.info("  E1 %-30s %6.3f %6.3f %6.3f %6d %5.1fs",
                 method, m["accuracy"], m["f1"], m["roc_auc"], m["n_rules"], m["training_time"])
    for method, m in all_results["E2"].items():
        log.info("  E2 %-30s %6.3f %6.3f %6.3f %6d %5.1fs",
                 method, m["accuracy"], m["f1"], m["roc_auc"], m["final_rules"], m["training_time"])
    for method, m in all_results["E3"].items():
        log.info("  E3 %-30s %6.3f %6.3f %6.3f %6d %5.1fs",
                 method, m["accuracy"], m["f1"], m["roc_auc"], m["n_rules"], m["training_time"])
    log.info("=" * 72)
