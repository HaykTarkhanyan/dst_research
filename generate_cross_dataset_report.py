"""
Generate a cross-dataset comparison report for the rule induction paper.

Creates a PDF with comprehensive comparisons across all datasets with experiment results.
Also exports LaTeX tables ready for paper inclusion.

Usage:
    python generate_cross_dataset_report.py
    python generate_cross_dataset_report.py --latex
    python generate_cross_dataset_report.py --output results_summary.pdf
"""
import argparse
import json
import os
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

RESULTS_BASE = "experiment_results"
DATASETS_DIR = os.path.join("src", "datasets")
SEED = 509

# Consistent color palette
COLORS = {
    "primary": "#4C72B0",
    "success": "#55A868",
    "danger": "#C44E52",
    "purple": "#8172B3",
    "yellow": "#CCB974",
    "cyan": "#64B5CD",
    "orange": "#DD8452",
    "pink": "#DA8BC3",
}
COLOR_LIST = list(COLORS.values())

# Dataset display names
DATASET_NAMES = {
    "breast-cancer-wisconsin": "Breast Cancer",
    "heart-disease": "Heart Disease",
    "ionosphere": "Ionosphere",
    "pima-diabetes": "PIMA Diabetes",
    "wine": "Wine",
}


def load_json(path):
    """Load JSON file, returning None if missing or invalid."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def safe_get(d, *keys, default=None):
    """Safely get nested dict values."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def fmt(val, decimals=3):
    """Format metric value."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def get_display_name(dataset):
    """Get human-readable dataset name."""
    return DATASET_NAMES.get(dataset, dataset.replace("-", " ").title())


def load_all_results():
    """Load results from all datasets."""
    results = {}
    for dataset in os.listdir(RESULTS_BASE):
        dataset_dir = os.path.join(RESULTS_BASE, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        
        results[dataset] = {
            "E1": load_json(os.path.join(dataset_dir, "E1_rule_source_comparison.json")),
            "E2": load_json(os.path.join(dataset_dir, "E2_iterative_refinement.json")),
            "E3": load_json(os.path.join(dataset_dir, "E3_ensemble.json")),
            "E4": load_json(os.path.join(dataset_dir, "E4_pruning_pareto.json")),
            "E5": load_json(os.path.join(dataset_dir, "E5_baselines.json")),
        }
    return results


# ============================================================================
# PAGE GENERATORS
# ============================================================================

def add_title_page(pdf, datasets, results):
    """Title page with overview statistics."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    
    ax.text(0.5, 0.75, "Cross-Dataset Comparison Report", fontsize=26, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.65, "Uncertainty-Driven Rule Induction for DSGD++", fontsize=16,
            ha="center", va="center", transform=ax.transAxes, color="#444")
    
    # Summary stats
    n_datasets = len(datasets)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    summary = f"Datasets: {n_datasets}\nSeed: {SEED}\nGenerated: {timestamp}"
    ax.text(0.5, 0.5, summary, fontsize=12, ha="center", va="center",
            transform=ax.transAxes, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))
    
    # Dataset list
    dataset_list = "\n".join([f"  • {get_display_name(d)}" for d in sorted(datasets)])
    ax.text(0.5, 0.3, f"Datasets:\n{dataset_list}", fontsize=11, ha="center", va="center",
            transform=ax.transAxes)
    
    pdf.savefig(fig)
    plt.close(fig)


def add_dataset_overview_page(pdf, datasets):
    """Overview of all datasets with sample counts and features."""
    import pandas as pd
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Dataset Overview", fontsize=14, fontweight="bold")
    
    # Collect dataset stats
    stats = []
    for dataset in sorted(datasets):
        path = os.path.join(DATASETS_DIR, f"{dataset}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            n_samples = len(df)
            n_features = len(df.columns) - 1
            y = df["labels"] if "labels" in df.columns else df.iloc[:, -1]
            class_counts = y.value_counts()
            balance = class_counts.min() / class_counts.max()
            stats.append({
                "dataset": get_display_name(dataset),
                "samples": n_samples,
                "features": n_features,
                "classes": len(class_counts),
                "balance": balance,
            })
    
    if not stats:
        plt.close(fig)
        return
    
    # Left: Bar chart of sample sizes
    names = [s["dataset"] for s in stats]
    samples = [s["samples"] for s in stats]
    features = [s["features"] for s in stats]
    
    x = np.arange(len(names))
    width = 0.35
    
    axes[0].bar(x - width/2, samples, width, label='Samples', color=COLORS["primary"])
    axes[0].bar(x + width/2, [f * 10 for f in features], width, label='Features (×10)', color=COLORS["success"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, ha="right")
    axes[0].legend()
    axes[0].set_title("Dataset Sizes")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add labels
    for i, (s, f) in enumerate(zip(samples, features)):
        axes[0].annotate(str(s), xy=(i - width/2, s), xytext=(0, 3),
                        textcoords="offset points", ha='center', fontsize=8)
        axes[0].annotate(str(f), xy=(i + width/2, f * 10), xytext=(0, 3),
                        textcoords="offset points", ha='center', fontsize=8)
    
    # Right: Summary table
    axes[1].axis("off")
    table_data = [["Dataset", "Samples", "Features", "Classes", "Balance"]]
    for s in stats:
        table_data.append([
            s["dataset"], str(s["samples"]), str(s["features"]),
            str(s["classes"]), f'{s["balance"]:.2f}'
        ])
    
    table = axes[1].table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for j in range(len(table_data[0])):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#f0f0f0")
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_master_results_table_page(pdf, datasets, results):
    """Master table: All methods × all datasets for E1."""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle("E1: Rule Source Comparison Across Datasets", fontsize=14, fontweight="bold")
    ax.axis("off")
    
    # Collect all methods
    all_methods = set()
    for dataset in datasets:
        e1 = results[dataset].get("E1", {})
        if e1:
            all_methods.update(e1.keys())
    all_methods = sorted(all_methods)
    
    # Build table: rows = methods, columns = datasets (Acc, F1, AUC for each)
    header = ["Method"] + [get_display_name(d)[:12] for d in sorted(datasets)] + ["Average"]
    table_data = [header]
    
    method_avgs = {}
    for method in all_methods:
        row = [method]
        accs = []
        for dataset in sorted(datasets):
            e1 = results[dataset].get("E1", {})
            if e1 and method in e1:
                acc = safe_get(e1[method], "accuracy", default=0)
                row.append(fmt(acc))
                if acc:
                    accs.append(acc)
            else:
                row.append("-")
        avg = np.mean(accs) if accs else 0
        method_avgs[method] = avg
        row.append(fmt(avg))
        table_data.append(row)
    
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    
    # Style header
    for j in range(len(header)):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#f0f0f0")
    
    # Highlight best per column (dataset)
    for col in range(1, len(header)):
        col_vals = []
        for row in range(1, len(table_data)):
            try:
                col_vals.append((row, float(table_data[row][col])))
            except ValueError:
                pass
        if col_vals:
            best_row = max(col_vals, key=lambda x: x[1])[0]
            table[best_row, col].set_facecolor("#d4edda")
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_accuracy_heatmap_page(pdf, datasets, results):
    """Heatmap of accuracy across methods and datasets."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("E1: Accuracy Heatmap", fontsize=14, fontweight="bold")
    
    # Collect all methods
    all_methods = set()
    for dataset in datasets:
        e1 = results[dataset].get("E1", {})
        if e1:
            all_methods.update(e1.keys())
    methods = sorted(all_methods)
    sorted_datasets = sorted(datasets)
    
    # Build matrix
    matrix = np.zeros((len(methods), len(sorted_datasets)))
    for i, method in enumerate(methods):
        for j, dataset in enumerate(sorted_datasets):
            e1 = results[dataset].get("E1", {})
            if e1 and method in e1:
                matrix[i, j] = safe_get(e1[method], "accuracy", default=0)
    
    # Plot heatmap
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.0)
    
    # Labels
    ax.set_xticks(np.arange(len(sorted_datasets)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels([get_display_name(d)[:12] for d in sorted_datasets])
    ax.set_yticklabels(methods)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(sorted_datasets)):
            val = matrix[i, j]
            color = "white" if val < 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Accuracy", rotation=-90, va="bottom")
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_f1_heatmap_page(pdf, datasets, results):
    """Heatmap of F1 score across methods and datasets."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("E1: F1 Score Heatmap", fontsize=14, fontweight="bold")
    
    # Collect all methods
    all_methods = set()
    for dataset in datasets:
        e1 = results[dataset].get("E1", {})
        if e1:
            all_methods.update(e1.keys())
    methods = sorted(all_methods)
    sorted_datasets = sorted(datasets)
    
    # Build matrix
    matrix = np.zeros((len(methods), len(sorted_datasets)))
    for i, method in enumerate(methods):
        for j, dataset in enumerate(sorted_datasets):
            e1 = results[dataset].get("E1", {})
            if e1 and method in e1:
                matrix[i, j] = safe_get(e1[method], "f1", default=0)
    
    # Plot heatmap
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.0)
    
    ax.set_xticks(np.arange(len(sorted_datasets)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels([get_display_name(d)[:12] for d in sorted_datasets])
    ax.set_yticklabels(methods)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    
    for i in range(len(methods)):
        for j in range(len(sorted_datasets)):
            val = matrix[i, j]
            color = "white" if val < 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("F1 Score", rotation=-90, va="bottom")
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_auc_heatmap_page(pdf, datasets, results):
    """Heatmap of AUC across methods and datasets."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("E1: ROC-AUC Heatmap", fontsize=14, fontweight="bold")
    
    all_methods = set()
    for dataset in datasets:
        e1 = results[dataset].get("E1", {})
        if e1:
            all_methods.update(e1.keys())
    methods = sorted(all_methods)
    sorted_datasets = sorted(datasets)
    
    matrix = np.zeros((len(methods), len(sorted_datasets)))
    for i, method in enumerate(methods):
        for j, dataset in enumerate(sorted_datasets):
            e1 = results[dataset].get("E1", {})
            if e1 and method in e1:
                matrix[i, j] = safe_get(e1[method], "roc_auc", default=0) or 0
    
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.0)
    
    ax.set_xticks(np.arange(len(sorted_datasets)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels([get_display_name(d)[:12] for d in sorted_datasets])
    ax.set_yticklabels(methods)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    
    for i in range(len(methods)):
        for j in range(len(sorted_datasets)):
            val = matrix[i, j]
            color = "white" if val < 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("ROC-AUC", rotation=-90, va="bottom")
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_average_improvement_page(pdf, datasets, results):
    """Bar chart showing average improvement by rule miner over baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Rule Miner Improvement Over Baseline (single_feature)", fontsize=14, fontweight="bold")
    
    # Calculate improvement for each miner
    miners = ["skope_rules", "ripper", "decision_tree"]
    improvements = {m: [] for m in miners}
    
    for dataset in datasets:
        e1 = results[dataset].get("E1", {})
        if not e1:
            continue
        baseline = safe_get(e1, "single_feature", "accuracy", default=0)
        if not baseline:
            continue
        for miner in miners:
            miner_acc = safe_get(e1, miner, "accuracy", default=0)
            if miner_acc:
                improvements[miner].append(miner_acc - baseline)
    
    # Left: Average improvement bar chart
    avg_impr = {m: np.mean(improvements[m]) if improvements[m] else 0 for m in miners}
    
    x = np.arange(len(miners))
    colors = [COLORS["success"] if avg_impr[m] > 0 else COLORS["danger"] for m in miners]
    bars = axes[0].bar(x, [avg_impr[m] for m in miners], color=colors)
    
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace("_", "\n") for m in miners])
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].set_ylabel("Average Accuracy Improvement")
    axes[0].set_title("Average Across Datasets")
    axes[0].grid(axis='y', alpha=0.3)
    
    for bar, m in zip(bars, miners):
        val = avg_impr[m]
        sign = "+" if val > 0 else ""
        axes[0].annotate(f'{sign}{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3 if val > 0 else -12), textcoords="offset points",
                        ha='center', fontsize=10, fontweight='bold')
    
    # Right: Per-dataset breakdown
    sorted_datasets = sorted(datasets)
    x = np.arange(len(sorted_datasets))
    width = 0.25
    
    for i, miner in enumerate(miners):
        impr_per_ds = []
        for dataset in sorted_datasets:
            e1 = results[dataset].get("E1", {})
            baseline = safe_get(e1, "single_feature", "accuracy", default=0)
            miner_acc = safe_get(e1, miner, "accuracy", default=0)
            if baseline and miner_acc:
                impr_per_ds.append(miner_acc - baseline)
            else:
                impr_per_ds.append(0)
        
        axes[1].bar(x + (i - 1) * width, impr_per_ds, width, 
                   label=miner.replace("_", " "), color=COLOR_LIST[i])
    
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([get_display_name(d)[:10] for d in sorted_datasets], rotation=15, ha="right")
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_ylabel("Accuracy Change")
    axes[1].set_title("Per-Dataset Breakdown")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', alpha=0.3)
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_dsgd_vs_baselines_page(pdf, datasets, results):
    """Compare best DSGD++ to best standalone baseline per dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("Best DSGD++ vs Best Baseline per Dataset", fontsize=14, fontweight="bold")
    
    comparisons = []
    for dataset in sorted(datasets):
        e1 = results[dataset].get("E1", {})
        e5 = results[dataset].get("E5", {})
        
        if not e1 or not e5:
            continue
        
        # Best E1
        best_e1_acc = 0
        best_e1_method = None
        for method, m in e1.items():
            acc = safe_get(m, "accuracy", default=0)
            if acc > best_e1_acc:
                best_e1_acc = acc
                best_e1_method = method
        
        # Best E5
        best_e5_acc = 0
        best_e5_method = None
        for method, m in e5.items():
            if "error" in m:
                continue
            acc = safe_get(m, "accuracy", default=0)
            if acc > best_e5_acc:
                best_e5_acc = acc
                best_e5_method = method
        
        if best_e1_method and best_e5_method:
            comparisons.append({
                "dataset": get_display_name(dataset),
                "e1_acc": best_e1_acc,
                "e1_method": best_e1_method,
                "e5_acc": best_e5_acc,
                "e5_method": best_e5_method,
            })
    
    if not comparisons:
        plt.close(fig)
        return
    
    # Left: Grouped bar chart
    x = np.arange(len(comparisons))
    width = 0.35
    
    e1_accs = [c["e1_acc"] for c in comparisons]
    e5_accs = [c["e5_acc"] for c in comparisons]
    
    bars1 = axes[0].bar(x - width/2, e1_accs, width, label='Best DSGD++', color=COLORS["success"])
    bars2 = axes[0].bar(x + width/2, e5_accs, width, label='Best Baseline', color=COLORS["primary"])
    
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c["dataset"][:10] for c in comparisons], rotation=15, ha="right")
    axes[0].set_ylim(0.5, 1.05)
    axes[0].legend()
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy Comparison")
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add delta annotations
    for i, c in enumerate(comparisons):
        delta = c["e1_acc"] - c["e5_acc"]
        color = COLORS["success"] if delta > 0 else COLORS["danger"]
        sign = "+" if delta > 0 else ""
        y_pos = max(c["e1_acc"], c["e5_acc"]) + 0.02
        axes[0].annotate(f'{sign}{delta:.2f}', xy=(i, y_pos), ha='center',
                        fontsize=9, fontweight='bold', color=color)
    
    # Right: Win/Loss summary
    axes[1].axis("off")
    
    wins = sum(1 for c in comparisons if c["e1_acc"] > c["e5_acc"])
    ties = sum(1 for c in comparisons if c["e1_acc"] == c["e5_acc"])
    losses = sum(1 for c in comparisons if c["e1_acc"] < c["e5_acc"])
    
    summary_text = f"""
═══ Win/Tie/Loss Summary ═══

DSGD++ Wins:  {wins}
Ties:         {ties}
Baseline Wins: {losses}

═══ Per-Dataset Details ═══
"""
    for c in comparisons:
        delta = c["e1_acc"] - c["e5_acc"]
        result = "WIN" if delta > 0 else "LOSS" if delta < 0 else "TIE"
        sign = "+" if delta > 0 else ""
        summary_text += f"\n{c['dataset'][:12]:12s}: {result} ({sign}{delta:.3f})"
        summary_text += f"\n  DSGD++: {c['e1_method']}"
        summary_text += f"\n  Base:   {c['e5_method']}\n"
    
    axes[1].text(0.1, 0.95, summary_text, transform=axes[1].transAxes, fontsize=9,
                va="top", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_rules_vs_accuracy_page(pdf, datasets, results):
    """Scatter plot: number of rules vs accuracy (interpretability-accuracy tradeoff)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Interpretability-Accuracy Tradeoff: Rules vs Accuracy", fontsize=14, fontweight="bold")
    
    # Collect all points
    for i, dataset in enumerate(sorted(datasets)):
        e1 = results[dataset].get("E1", {})
        if not e1:
            continue
        
        for method, m in e1.items():
            acc = safe_get(m, "accuracy", default=0)
            rules = safe_get(m, "n_rules", default=0)
            if acc and rules:
                color = COLOR_LIST[i % len(COLOR_LIST)]
                marker = ['o', 's', '^', 'D'][list(e1.keys()).index(method) % 4]
                ax.scatter(rules, acc, c=color, marker=marker, s=80, alpha=0.7,
                          label=f"{get_display_name(dataset)[:8]}-{method[:8]}")
    
    ax.set_xlabel("Number of Rules")
    ax.set_ylabel("Accuracy")
    ax.set_title("Each point is a (dataset, method) combination")
    ax.grid(alpha=0.3)
    
    # Add trend line
    all_rules = []
    all_accs = []
    for dataset in datasets:
        e1 = results[dataset].get("E1", {})
        if e1:
            for m in e1.values():
                r = safe_get(m, "n_rules", default=0)
                a = safe_get(m, "accuracy", default=0)
                if r and a:
                    all_rules.append(r)
                    all_accs.append(a)
    
    if len(all_rules) > 2:
        z = np.polyfit(all_rules, all_accs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(all_rules), max(all_rules), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.5, label=f"Trend (slope={z[0]:.4f})")
    
    # Simplified legend
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 8:
        ax.legend(handles[:8], labels[:8], fontsize=7, loc="lower right", ncol=2)
    else:
        ax.legend(fontsize=8, loc="lower right")
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_radar_chart_page(pdf, datasets, results):
    """Radar chart: multi-metric comparison of methods (averaged across datasets)."""
    from math import pi
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.suptitle("Multi-Metric Radar: Methods Averaged Across Datasets", fontsize=14, fontweight="bold", y=0.98)
    
    # Collect all methods
    all_methods = set()
    for dataset in datasets:
        e1 = results[dataset].get("E1", {})
        if e1:
            all_methods.update(e1.keys())
    methods = sorted(all_methods)
    
    metrics = ["Accuracy", "F1", "AUC", "Speed"]  # Speed is inverted training time
    
    # Calculate averages per method
    method_scores = {}
    for method in methods:
        scores = {"Accuracy": [], "F1": [], "AUC": [], "Speed": []}
        for dataset in datasets:
            e1 = results[dataset].get("E1", {})
            if e1 and method in e1:
                m = e1[method]
                if safe_get(m, "accuracy"):
                    scores["Accuracy"].append(safe_get(m, "accuracy"))
                if safe_get(m, "f1"):
                    scores["F1"].append(safe_get(m, "f1"))
                if safe_get(m, "roc_auc"):
                    scores["AUC"].append(safe_get(m, "roc_auc"))
                if safe_get(m, "training_time"):
                    # Invert and normalize speed (faster = higher score)
                    scores["Speed"].append(1 / (1 + safe_get(m, "training_time") / 10))
        
        method_scores[method] = {k: np.mean(v) if v else 0 for k, v in scores.items()}
    
    # Radar plot
    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]  # Close the loop
    
    for i, method in enumerate(methods):
        values = [method_scores[method][m] for m in metrics]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=COLOR_LIST[i % len(COLOR_LIST)])
        ax.fill(angles, values, alpha=0.15, color=COLOR_LIST[i % len(COLOR_LIST)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    
    fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def add_training_time_comparison_page(pdf, datasets, results):
    """Training time comparison across datasets and methods."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Training Time Analysis", fontsize=14, fontweight="bold")
    
    # Left: Stacked bar by dataset
    all_methods = set()
    for dataset in datasets:
        e1 = results[dataset].get("E1", {})
        if e1:
            all_methods.update(e1.keys())
    methods = sorted(all_methods)
    sorted_datasets = sorted(datasets)
    
    x = np.arange(len(sorted_datasets))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        times = []
        for dataset in sorted_datasets:
            e1 = results[dataset].get("E1", {})
            t = safe_get(e1, method, "training_time", default=0) if e1 else 0
            times.append(t)
        
        axes[0].bar(x + (i - len(methods)/2 + 0.5) * width, times, width,
                   label=method, color=COLOR_LIST[i % len(COLOR_LIST)])
    
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([get_display_name(d)[:10] for d in sorted_datasets], rotation=15, ha="right")
    axes[0].set_ylabel("Training Time (s)")
    axes[0].set_title("Training Time by Dataset")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Right: Average time per method
    avg_times = {}
    for method in methods:
        times = []
        for dataset in datasets:
            e1 = results[dataset].get("E1", {})
            if e1 and method in e1:
                t = safe_get(e1[method], "training_time")
                if t:
                    times.append(t)
        avg_times[method] = np.mean(times) if times else 0
    
    sorted_methods = sorted(avg_times.keys(), key=lambda m: avg_times[m], reverse=True)
    
    y_pos = np.arange(len(sorted_methods))
    bars = axes[1].barh(y_pos, [avg_times[m] for m in sorted_methods],
                        color=[COLOR_LIST[methods.index(m) % len(COLOR_LIST)] for m in sorted_methods])
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(sorted_methods)
    axes[1].set_xlabel("Average Training Time (s)")
    axes[1].set_title("Average Time per Method")
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].invert_yaxis()
    
    for bar, m in zip(bars, sorted_methods):
        axes[1].annotate(f'{avg_times[m]:.1f}s', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                        xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=8)
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_ranking_analysis_page(pdf, datasets, results):
    """Analyze which method ranks best on average across datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Ranking Analysis Across Datasets", fontsize=14, fontweight="bold")
    
    # Collect all methods
    all_methods = set()
    for dataset in datasets:
        e1 = results[dataset].get("E1", {})
        if e1:
            all_methods.update(e1.keys())
    methods = sorted(all_methods)
    
    # Calculate ranks per dataset
    ranks = {m: [] for m in methods}
    for dataset in datasets:
        e1 = results[dataset].get("E1", {})
        if not e1:
            continue
        
        # Get accuracies and rank them
        accs = [(m, safe_get(e1.get(m, {}), "accuracy", default=0)) for m in methods if m in e1]
        accs.sort(key=lambda x: -x[1])  # Higher is better
        
        for rank, (method, _) in enumerate(accs, 1):
            ranks[method].append(rank)
    
    # Left: Average rank bar chart
    avg_ranks = {m: np.mean(ranks[m]) if ranks[m] else len(methods) for m in methods}
    sorted_by_rank = sorted(methods, key=lambda m: avg_ranks[m])
    
    y_pos = np.arange(len(sorted_by_rank))
    colors = [COLOR_LIST[i % len(COLOR_LIST)] for i in range(len(sorted_by_rank))]
    bars = axes[0].barh(y_pos, [avg_ranks[m] for m in sorted_by_rank], color=colors)
    
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(sorted_by_rank)
    axes[0].set_xlabel("Average Rank (lower is better)")
    axes[0].set_title("Average Rank Across Datasets")
    axes[0].grid(axis='x', alpha=0.3)
    
    for bar, m in zip(bars, sorted_by_rank):
        axes[0].annotate(f'{avg_ranks[m]:.2f}', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                        xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=9)
    
    # Right: Rank distribution table
    axes[1].axis("off")
    
    header = ["Method"] + [f"Rank {i}" for i in range(1, len(methods) + 1)] + ["Avg"]
    table_data = [header]
    
    for method in sorted_by_rank:
        row = [method]
        rank_counts = [0] * len(methods)
        for r in ranks[method]:
            if r <= len(methods):
                rank_counts[r - 1] += 1
        row.extend([str(c) if c > 0 else "-" for c in rank_counts])
        row.append(f"{avg_ranks[method]:.2f}")
        table_data.append(row)
    
    table = axes[1].table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    for j in range(len(header)):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#f0f0f0")
    
    # Highlight first place counts
    for i, method in enumerate(sorted_by_rank, 1):
        if ranks[method].count(1) > 0:
            table[i, 1].set_facecolor("#d4edda")
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_baselines_comparison_page(pdf, datasets, results):
    """Comprehensive comparison of E5 baselines across datasets."""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle("E5: Standalone Baselines Across Datasets", fontsize=14, fontweight="bold")
    ax.axis("off")
    
    # Collect all baseline methods
    all_baselines = set()
    for dataset in datasets:
        e5 = results[dataset].get("E5", {})
        if e5:
            all_baselines.update(k for k, v in e5.items() if "error" not in v)
    baselines = sorted(all_baselines)
    sorted_datasets = sorted(datasets)
    
    # Build table
    header = ["Baseline"] + [get_display_name(d)[:10] for d in sorted_datasets] + ["Average"]
    table_data = [header]
    
    for baseline in baselines:
        row = [baseline]
        accs = []
        for dataset in sorted_datasets:
            e5 = results[dataset].get("E5", {})
            if e5 and baseline in e5 and "error" not in e5[baseline]:
                acc = safe_get(e5[baseline], "accuracy", default=0)
                row.append(fmt(acc))
                if acc:
                    accs.append(acc)
            else:
                row.append("-")
        avg = np.mean(accs) if accs else 0
        row.append(fmt(avg))
        table_data.append(row)
    
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    for j in range(len(header)):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#f0f0f0")
    
    # Highlight best per column
    for col in range(1, len(header)):
        col_vals = []
        for row in range(1, len(table_data)):
            try:
                col_vals.append((row, float(table_data[row][col])))
            except ValueError:
                pass
        if col_vals:
            best_row = max(col_vals, key=lambda x: x[1])[0]
            table[best_row, col].set_facecolor("#d4edda")
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_key_findings_page(pdf, datasets, results):
    """Summary page with key paper findings."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    fig.suptitle("Key Findings Summary", fontsize=16, fontweight="bold")
    
    findings = []
    findings.append("═══════════════════════════════════════════════════════")
    findings.append("                   KEY FINDINGS FOR PAPER")
    findings.append("═══════════════════════════════════════════════════════\n")
    
    # 1. Best overall method
    method_accs = defaultdict(list)
    for dataset in datasets:
        e1 = results[dataset].get("E1", {})
        if e1:
            for method, m in e1.items():
                acc = safe_get(m, "accuracy", default=0)
                if acc:
                    method_accs[method].append(acc)
    
    avg_accs = {m: np.mean(accs) for m, accs in method_accs.items()}
    if avg_accs:
        best_method = max(avg_accs, key=avg_accs.get)
        findings.append(f"1. BEST E1 METHOD (avg across datasets):")
        findings.append(f"   {best_method} with {avg_accs[best_method]:.3f} accuracy\n")
    
    # 2. DSGD++ vs baselines
    dsgd_wins = 0
    baseline_wins = 0
    for dataset in datasets:
        e1 = results[dataset].get("E1", {})
        e5 = results[dataset].get("E5", {})
        if e1 and e5:
            best_e1 = max([safe_get(m, "accuracy", default=0) for m in e1.values()])
            best_e5 = max([safe_get(m, "accuracy", default=0) for m in e5.values() if "error" not in m], default=0)
            if best_e1 > best_e5:
                dsgd_wins += 1
            elif best_e5 > best_e1:
                baseline_wins += 1
    
    findings.append(f"2. DSGD++ vs BASELINES:")
    findings.append(f"   DSGD++ wins: {dsgd_wins} datasets")
    findings.append(f"   Baselines win: {baseline_wins} datasets\n")
    
    # 3. Best rule miner improvement
    miner_improvement = {}
    for miner in ["skope_rules", "ripper", "decision_tree"]:
        improvements = []
        for dataset in datasets:
            e1 = results[dataset].get("E1", {})
            if e1:
                baseline = safe_get(e1, "single_feature", "accuracy", default=0)
                miner_acc = safe_get(e1, miner, "accuracy", default=0)
                if baseline and miner_acc:
                    improvements.append(miner_acc - baseline)
        if improvements:
            miner_improvement[miner] = np.mean(improvements)
    
    if miner_improvement:
        best_miner = max(miner_improvement, key=miner_improvement.get)
        findings.append(f"3. BEST RULE MINER:")
        findings.append(f"   {best_miner} with +{miner_improvement[best_miner]:.3f} avg improvement")
        findings.append(f"   (over single_feature baseline)\n")
    
    # 4. Dataset hardness ranking
    dataset_best_acc = {}
    for dataset in datasets:
        e1 = results[dataset].get("E1", {})
        if e1:
            best_acc = max([safe_get(m, "accuracy", default=0) for m in e1.values()])
            dataset_best_acc[dataset] = best_acc
    
    if dataset_best_acc:
        sorted_datasets = sorted(dataset_best_acc.keys(), key=lambda d: dataset_best_acc[d])
        findings.append(f"4. DATASET DIFFICULTY (hardest to easiest):")
        for i, d in enumerate(sorted_datasets, 1):
            findings.append(f"   {i}. {get_display_name(d)}: {dataset_best_acc[d]:.3f}")
        findings.append("")
    
    # 5. Rule count analysis
    findings.append(f"5. INTERPRETABILITY:")
    for dataset in datasets:
        e1 = results[dataset].get("E1", {})
        if e1:
            single = safe_get(e1, "single_feature", "n_rules", default=0)
            skope = safe_get(e1, "skope_rules", "n_rules", default=0)
            findings.append(f"   {get_display_name(dataset)}: {single} base → {skope} with SkopeRules")
    
    ax.text(0.05, 0.95, "\n".join(findings), transform=ax.transAxes, fontsize=10,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))
    
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_full_results_table_page(pdf, datasets, results):
    """Comprehensive table with all metrics for E1."""
    # This might need multiple pages for many datasets
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.suptitle("Complete E1 Results: All Metrics", fontsize=14, fontweight="bold")
    ax.axis("off")
    
    # Collect all methods
    all_methods = set()
    for dataset in datasets:
        e1 = results[dataset].get("E1", {})
        if e1:
            all_methods.update(e1.keys())
    methods = sorted(all_methods)
    
    # Build comprehensive table
    rows = []
    rows.append(["Dataset", "Method", "Acc", "F1", "Prec", "Recall", "AUC", "Rules", "Time(s)"])
    
    for dataset in sorted(datasets):
        e1 = results[dataset].get("E1", {})
        if not e1:
            continue
        
        first_row = True
        for method in methods:
            if method not in e1:
                continue
            m = e1[method]
            row = [
                get_display_name(dataset)[:12] if first_row else "",
                method,
                fmt(safe_get(m, "accuracy")),
                fmt(safe_get(m, "f1")),
                fmt(safe_get(m, "precision")),
                fmt(safe_get(m, "recall")),
                fmt(safe_get(m, "roc_auc")),
                str(safe_get(m, "n_rules", default="-")),
                fmt(safe_get(m, "training_time"), 1) if safe_get(m, "training_time") else "-",
            ]
            rows.append(row)
            first_row = False
    
    table = ax.table(cellText=rows, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.2)
    
    for j in range(len(rows[0])):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#f0f0f0")
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def export_latex_tables(datasets, results, output_path):
    """Export publication-ready LaTeX tables."""
    lines = [
        "% Cross-Dataset LaTeX Tables for Rule Induction Paper",
        f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "% Datasets: " + ", ".join(sorted(datasets)),
        "",
        "\\usepackage{booktabs}",
        "",
    ]
    
    # Table 1: Main results - E1 accuracy
    lines.append("% Table 1: E1 Accuracy across datasets")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{DSGD++ with Different Rule Sources: Accuracy}")
    lines.append("\\label{tab:e1_accuracy}")
    
    all_methods = set()
    for dataset in datasets:
        e1 = results[dataset].get("E1", {})
        if e1:
            all_methods.update(e1.keys())
    methods = sorted(all_methods)
    sorted_datasets = sorted(datasets)
    
    col_spec = "l" + "c" * len(sorted_datasets) + "c"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    
    header = "Method & " + " & ".join([get_display_name(d)[:8] for d in sorted_datasets]) + " & Avg \\\\"
    lines.append(header)
    lines.append("\\midrule")
    
    # Find best per column for bolding
    best_per_col = {}
    for j, dataset in enumerate(sorted_datasets):
        e1 = results[dataset].get("E1", {})
        if e1:
            best_acc = max([safe_get(m, "accuracy", default=0) for m in e1.values()])
            best_per_col[j] = best_acc
    
    for method in methods:
        row = [f"DSGD++ + {method}"]
        accs = []
        for j, dataset in enumerate(sorted_datasets):
            e1 = results[dataset].get("E1", {})
            if e1 and method in e1:
                acc = safe_get(e1[method], "accuracy", default=0)
                if acc == best_per_col.get(j, -1):
                    row.append(f"\\textbf{{{acc:.3f}}}")
                else:
                    row.append(f"{acc:.3f}")
                if acc:
                    accs.append(acc)
            else:
                row.append("-")
        avg = np.mean(accs) if accs else 0
        row.append(f"{avg:.3f}")
        lines.append(" & ".join(row) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    
    # Table 2: DSGD++ vs Baselines
    lines.append("% Table 2: Best DSGD++ vs Best Baseline")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Best DSGD++ vs Best Standalone Baseline}")
    lines.append("\\label{tab:dsgd_vs_baseline}")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("Dataset & Best DSGD++ & Acc & Best Baseline & Acc & $\\Delta$ \\\\")
    lines.append("\\midrule")
    
    for dataset in sorted_datasets:
        e1 = results[dataset].get("E1", {})
        e5 = results[dataset].get("E5", {})
        
        if not e1 or not e5:
            continue
        
        best_e1_acc = 0
        best_e1_method = ""
        for method, m in e1.items():
            acc = safe_get(m, "accuracy", default=0)
            if acc > best_e1_acc:
                best_e1_acc = acc
                best_e1_method = method
        
        best_e5_acc = 0
        best_e5_method = ""
        for method, m in e5.items():
            if "error" in m:
                continue
            acc = safe_get(m, "accuracy", default=0)
            if acc > best_e5_acc:
                best_e5_acc = acc
                best_e5_method = method
        
        delta = best_e1_acc - best_e5_acc
        sign = "+" if delta >= 0 else ""
        
        lines.append(f"{get_display_name(dataset)} & {best_e1_method} & {best_e1_acc:.3f} & "
                    f"{best_e5_method} & {best_e5_acc:.3f} & {sign}{delta:.3f} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    
    # Table 3: Average improvement by miner
    lines.append("% Table 3: Average Improvement by Rule Miner")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Average Accuracy Improvement Over Single-Feature Baseline}")
    lines.append("\\label{tab:miner_improvement}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Rule Miner & Avg $\\Delta$ Accuracy & Datasets \\\\")
    lines.append("\\midrule")
    
    for miner in ["skope_rules", "ripper", "decision_tree"]:
        improvements = []
        for dataset in datasets:
            e1 = results[dataset].get("E1", {})
            if e1:
                baseline = safe_get(e1, "single_feature", "accuracy", default=0)
                miner_acc = safe_get(e1, miner, "accuracy", default=0)
                if baseline and miner_acc:
                    improvements.append(miner_acc - baseline)
        
        if improvements:
            avg = np.mean(improvements)
            sign = "+" if avg >= 0 else ""
            lines.append(f"{miner.replace('_', ' ').title()} & {sign}{avg:.3f} & {len(improvements)} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"LaTeX tables saved: {output_path}")


def generate_cross_dataset_report(output_path="experiment_results/cross_dataset_report.pdf", export_latex=False):
    """Generate the comprehensive cross-dataset comparison report."""
    results = load_all_results()
    datasets = list(results.keys())
    
    if not datasets:
        print("No experiment results found!")
        return
    
    print(f"Found {len(datasets)} datasets: {', '.join(datasets)}")
    
    with PdfPages(output_path) as pdf:
        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'Cross-Dataset Comparison Report'
        d['Author'] = 'DSGD++ Experiment Runner'
        d['Subject'] = 'Uncertainty-Driven Rule Induction'
        d['CreationDate'] = datetime.now()
        
        # Generate all pages
        print("Generating pages...")
        
        # 1. Title and overview
        add_title_page(pdf, datasets, results)
        add_dataset_overview_page(pdf, datasets)
        
        # 2. Key findings summary (good for paper abstract)
        add_key_findings_page(pdf, datasets, results)
        
        # 3. Master results tables
        add_master_results_table_page(pdf, datasets, results)
        add_full_results_table_page(pdf, datasets, results)
        
        # 4. Heatmaps
        add_accuracy_heatmap_page(pdf, datasets, results)
        add_f1_heatmap_page(pdf, datasets, results)
        add_auc_heatmap_page(pdf, datasets, results)
        
        # 5. Improvement analysis
        add_average_improvement_page(pdf, datasets, results)
        
        # 6. DSGD++ vs baselines
        add_dsgd_vs_baselines_page(pdf, datasets, results)
        add_baselines_comparison_page(pdf, datasets, results)
        
        # 7. Interpretability-accuracy tradeoff
        add_rules_vs_accuracy_page(pdf, datasets, results)
        
        # 8. Multi-metric analysis
        add_radar_chart_page(pdf, datasets, results)
        
        # 9. Training time
        add_training_time_comparison_page(pdf, datasets, results)
        
        # 10. Ranking analysis
        add_ranking_analysis_page(pdf, datasets, results)
    
    print(f"Report saved: {output_path}")
    
    # Export LaTeX if requested
    if export_latex:
        latex_path = output_path.replace(".pdf", "_tables.tex")
        export_latex_tables(datasets, results, latex_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cross-dataset comparison report")
    parser.add_argument("--output", "-o", type=str, 
                        default="experiment_results/cross_dataset_report.pdf",
                        help="Output PDF path")
    parser.add_argument("--latex", action="store_true",
                        help="Also export LaTeX tables")
    args = parser.parse_args()
    
    generate_cross_dataset_report(args.output, export_latex=args.latex)
