"""
Generate a PDF report with matplotlib figures and tables from experiment results.

Usage:
    python generate_report.py heart-disease
    python generate_report.py ionosphere
    python generate_report.py all    # generates for all datasets with results
    python generate_report.py heart-disease --latex  # also export LaTeX tables
    python generate_report.py heart-disease --experiments E1,E5  # specific experiments only
"""
import argparse
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


RESULTS_BASE = "experiment_results"
DATASETS_DIR = os.path.join("src", "datasets")
SEED = 509  # Match experiment seed for documentation

# Consistent color palette across all charts
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


def load_json(path):
    """Load JSON file, returning None if missing or invalid."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: could not load {path}: {e}")
        return None


def safe_get(d, *keys, default=None):
    """Safely get nested dict values without KeyError."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def fmt_metric(val, decimals=3):
    """Format a metric value, handling None/missing gracefully."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def add_title_page(pdf, dataset_name, e1=None, e5=None):
    """Title page with dataset info, timestamp, and summary stats."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    
    # Title
    ax.text(0.5, 0.75, "Rule Induction Experiment Report", fontsize=24, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.65, f"Dataset: {dataset_name}", fontsize=18,
            ha="center", va="center", transform=ax.transAxes, color="#444")
    ax.text(0.5, 0.55, "DSGD++ with Uncertainty-Driven Rule Mining", fontsize=14,
            ha="center", va="center", transform=ax.transAxes, color="#666")
    
    # Timestamp and metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    ax.text(0.5, 0.42, f"Generated: {timestamp}  |  Seed: {SEED}", fontsize=10,
            ha="center", va="center", transform=ax.transAxes, color="#888")
    
    # Quick summary stats if available
    summary_lines = []
    if e1:
        best_e1 = max(e1.items(), key=lambda x: safe_get(x[1], "accuracy", default=0))
        summary_lines.append(f"Best E1 (Rule Source): {best_e1[0]} — Acc: {fmt_metric(safe_get(best_e1[1], 'accuracy'))}")
    if e5:
        valid_e5 = {k: v for k, v in e5.items() if "error" not in v}
        if valid_e5:
            best_e5 = max(valid_e5.items(), key=lambda x: safe_get(x[1], "accuracy", default=0))
            summary_lines.append(f"Best E5 (Baseline): {best_e5[0]} — Acc: {fmt_metric(safe_get(best_e5[1], 'accuracy'))}")
    
    if summary_lines:
        ax.text(0.5, 0.28, "\n".join(summary_lines), fontsize=11,
                ha="center", va="center", transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))
    
    # Experiments included
    experiments = []
    if e1: experiments.append("E1: Rule Source Comparison")
    if e5: experiments.append("E5: Interpretable Baselines")
    if experiments:
        ax.text(0.5, 0.12, "Experiments: " + ", ".join(experiments), fontsize=9,
                ha="center", va="center", transform=ax.transAxes, color="#666")
    
    pdf.savefig(fig)
    plt.close(fig)


def add_dataset_stats_page(pdf, dataset_name):
    """Dataset statistics page with sample counts, features, and class distribution."""
    dataset_path = os.path.join(DATASETS_DIR, f"{dataset_name}.csv")
    if not os.path.exists(dataset_path):
        return
    
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Warning: could not load dataset {dataset_path}: {e}")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
    fig.suptitle(f"Dataset Overview — {dataset_name}", fontsize=14, fontweight="bold")
    
    # Stats summary (left panel)
    axes[0].axis("off")
    n_samples = len(df)
    n_features = len(df.columns) - 1  # exclude labels
    y = df["labels"] if "labels" in df.columns else df.iloc[:, -1]
    class_counts = y.value_counts().sort_index()
    n_classes = len(class_counts)
    
    # Calculate class balance
    majority_pct = class_counts.max() / n_samples * 100
    minority_pct = class_counts.min() / n_samples * 100
    
    stats_text = (
        f"Samples: {n_samples}\n"
        f"Features: {n_features}\n"
        f"Classes: {n_classes}\n"
        f"Train/Test: 70/30\n"
        f"Seed: {SEED}\n\n"
        f"Class Balance:\n"
        f"  Majority: {majority_pct:.1f}%\n"
        f"  Minority: {minority_pct:.1f}%"
    )
    axes[0].text(0.1, 0.9, stats_text, fontsize=11, va="top", ha="left",
                 transform=axes[0].transAxes, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))
    axes[0].set_title("Summary Statistics", fontsize=11)
    
    # Class distribution bar chart (middle panel)
    bars = axes[1].bar(class_counts.index.astype(str), class_counts.values, 
                       color=[COLORS["primary"], COLORS["success"]][:n_classes])
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Class Distribution", fontsize=11)
    # Add count labels on bars
    for bar, count in zip(bars, class_counts.values):
        axes[1].annotate(str(count), xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)
    
    # Feature names list (right panel)
    axes[2].axis("off")
    feature_names = [c for c in df.columns if c != "labels"]
    if len(feature_names) > 15:
        display_names = feature_names[:12] + ["...", f"({len(feature_names)} total)"]
    else:
        display_names = feature_names
    
    features_text = "Features:\n" + "\n".join(f"  {i+1}. {name[:25]}" for i, name in enumerate(display_names))
    axes[2].text(0.05, 0.95, features_text, fontsize=9, va="top", ha="left",
                 transform=axes[2].transAxes, family="monospace")
    axes[2].set_title("Feature List", fontsize=11)
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_training_time_page(pdf, e1, e5, dataset_name):
    """Training time comparison across methods."""
    if not e1 and not e5:
        return
    
    # Collect timing data
    methods = []
    times = []
    types = []
    
    if e1:
        for method, m in e1.items():
            t = safe_get(m, "training_time")
            if t is not None:
                methods.append(f"DSGD++ + {method}")
                times.append(t)
                types.append("E1")
    
    if e5:
        for method, m in e5.items():
            if "error" not in m:
                t = safe_get(m, "training_time")
                if t is not None:
                    methods.append(method)
                    times.append(t)
                    types.append("E5")
    
    if not methods:
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"Training Time Comparison — {dataset_name}", fontsize=14, fontweight="bold")
    
    # Sort by time
    sorted_data = sorted(zip(methods, times, types), key=lambda x: x[1], reverse=True)
    methods, times, types = zip(*sorted_data)
    
    # Color by type
    colors = [COLORS["success"] if t == "E1" else COLORS["primary"] for t in types]
    
    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, times, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Methods Ranked by Training Time")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()  # fastest at top
    
    # Add time labels
    for bar, t in zip(bars, times):
        ax.annotate(f'{t:.1f}s', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=8)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS["success"], label="DSGD++ (E1)"),
                       Patch(facecolor=COLORS["primary"], label="Baseline (E5)")]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_e1_page(pdf, data, dataset_name):
    """E1: Rule source comparison — bar chart + table."""
    if not data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(f"E1: Rule Source Comparison — {dataset_name}", fontsize=14, fontweight="bold")

    methods = list(data.keys())
    accs = [safe_get(data[m], "accuracy", default=0) for m in methods]
    f1s = [safe_get(data[m], "f1", default=0) for m in methods]
    aucs = [safe_get(data[m], "roc_auc", default=0) for m in methods]
    rules = [safe_get(data[m], "n_rules", default=0) for m in methods]

    # Bar chart: grouped by metric, legend shows models
    metrics = ["Accuracy", "F1", "AUC"]
    metric_values = [accs, f1s, aucs]
    x = np.arange(len(metrics))  # 3 metric groups
    n_methods = len(methods)
    
    # Calculate bar width to fit all methods with spacing
    total_width = 0.7  # total width for all bars in a group
    bar_width = total_width / n_methods
    
    # Plot bars for each method
    bars_list = []
    for i, method in enumerate(methods):
        offset = (i - (n_methods - 1) / 2) * bar_width
        vals = [accs[i], f1s[i], aucs[i]]
        bars = axes[0].bar(x + offset, vals, bar_width * 0.9, 
                           label=method, color=COLOR_LIST[i % len(COLOR_LIST)])
        bars_list.append((bars, vals))
    
    # Add value labels on bars (staggered heights to avoid overlap)
    for i, (bars, vals) in enumerate(bars_list):
        for j, (bar, val) in enumerate(zip(bars, vals)):
            height = bar.get_height()
            # Stagger vertical offset based on method index for overlapping bars
            y_offset = 0.02 + (i % 2) * 0.03  # alternate between 0.02 and 0.05
            axes[0].annotate(f'{val:.2f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3 + (i % 2) * 8),  # stagger text position
                             textcoords="offset points",
                             ha='center', va='bottom', fontsize=7, rotation=0)
    
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, fontsize=10)
    axes[0].set_ylim(0, 1.15)  # extra space for labels
    axes[0].legend(fontsize=8, loc='upper right', framealpha=0.9)
    axes[0].set_title("Metrics by Method")
    axes[0].set_ylabel("Score")
    axes[0].grid(axis="y", alpha=0.3)

    # Table
    axes[1].axis("off")
    table_data = [["Method", "Acc", "F1", "AUC", "Rules"]]
    for m in methods:
        d = data[m]
        table_data.append([
            m, 
            fmt_metric(safe_get(d, "accuracy")), 
            fmt_metric(safe_get(d, "f1")),
            fmt_metric(safe_get(d, "roc_auc")), 
            str(safe_get(d, "n_rules", default="-"))
        ])

    table = axes[1].table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    # Bold header
    for j in range(len(table_data[0])):
        table[0, j].set_text_props(fontweight="bold")
    # Highlight best accuracy row
    if accs:
        best_idx = np.argmax(accs) + 1
        for j in range(len(table_data[0])):
            table[best_idx, j].set_facecolor("#d4edda")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_e2_page(pdf, data, dataset_name):
    """E2: Iterative refinement — rule growth + metrics."""
    if not data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"E2: Iterative Uncertainty-Guided Refinement — {dataset_name}",
                 fontsize=14, fontweight="bold")

    # Table
    axes[0].axis("off")
    table_data = [["Miner", "Acc", "F1", "AUC", "Init Rules", "Final Rules", "Iters"]]
    for miner, m in data.items():
        table_data.append([
            miner, 
            fmt_metric(safe_get(m, "accuracy")), 
            fmt_metric(safe_get(m, "f1")), 
            fmt_metric(safe_get(m, "roc_auc")),
            str(safe_get(m, "initial_rules", default="-")), 
            str(safe_get(m, "final_rules", default="-")), 
            str(safe_get(m, "iterations", default="-")),
        ])
    table = axes[0].table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for j in range(len(table_data[0])):
        table[0, j].set_text_props(fontweight="bold")
    axes[0].set_title("Results", fontsize=11)

    # Rule growth per iteration
    has_history = False
    for miner, m in data.items():
        history = safe_get(m, "history", default=[])
        if history:
            has_history = True
            iters = [0] + [safe_get(h, "iteration", default=i+1) for i, h in enumerate(history)]
            rule_counts = [safe_get(m, "initial_rules", default=0)] + [safe_get(h, "total_rules", default=0) for h in history]
            axes[1].plot(iters, rule_counts, marker="o", label=miner)
    
    if has_history:
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Total Rules")
        axes[1].set_title("Rule Growth per Iteration")
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.3)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No iteration history available", ha="center", va="center",
                     fontsize=11, color="#888", transform=axes[1].transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_method_vs_dst_page(pdf, e1, e5, dataset_name):
    """Compare baseline methods vs the same methods enhanced with DSGD++."""
    if not e1 or not e5:
        return
    
    # Define matching pairs: (E5 baseline name, E1 DSGD++ name, display name)
    method_pairs = [
        ("SkopeRules", "skope_rules", "SkopeRules"),
        ("DecisionTree_d4", "decision_tree", "DecisionTree"),
        ("RIPPER", "ripper", "RIPPER"),
    ]
    
    # Collect matched data
    comparisons = []
    for e5_name, e1_name, display_name in method_pairs:
        if e5_name in e5 and e1_name in e1:
            e5_data = e5[e5_name]
            e1_data = e1[e1_name]
            if "error" not in e5_data:
                comparisons.append({
                    "name": display_name,
                    "baseline_acc": safe_get(e5_data, "accuracy", default=0),
                    "baseline_f1": safe_get(e5_data, "f1", default=0),
                    "baseline_auc": safe_get(e5_data, "roc_auc", default=0) or 0,
                    "dst_acc": safe_get(e1_data, "accuracy", default=0),
                    "dst_f1": safe_get(e1_data, "f1", default=0),
                    "dst_auc": safe_get(e1_data, "roc_auc", default=0),
                })
    
    if not comparisons:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(f"Baseline vs DSGD++ Enhancement — {dataset_name}", fontsize=14, fontweight="bold")
    
    # Left: Grouped bar chart
    methods = [c["name"] for c in comparisons]
    x = np.arange(len(methods))
    width = 0.35
    
    # Accuracy comparison
    baseline_accs = [c["baseline_acc"] for c in comparisons]
    dst_accs = [c["dst_acc"] for c in comparisons]
    
    bars1 = axes[0].bar(x - width/2, baseline_accs, width, label='Baseline', color=COLORS["primary"])
    bars2 = axes[0].bar(x + width/2, dst_accs, width, label='+ DSGD++', color=COLORS["success"])
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        axes[0].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # Add delta arrows/text
    for i, (b_acc, d_acc) in enumerate(zip(baseline_accs, dst_accs)):
        delta = d_acc - b_acc
        color = COLORS["success"] if delta > 0 else COLORS["danger"]
        sign = "+" if delta > 0 else ""
        axes[0].annotate(f'{sign}{delta:.2f}', xy=(x[i], max(b_acc, d_acc) + 0.08),
                        ha='center', fontsize=9, fontweight='bold', color=color)
    
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy: Method vs Method + DSGD++')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].legend(loc='lower right')
    axes[0].set_ylim(0, 1.15)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Right: Table with all metrics + delta
    axes[1].axis("off")
    table_data = [["Method", "Metric", "Baseline", "+ DSGD++", "Δ"]]
    
    for c in comparisons:
        # Accuracy row
        delta_acc = c["dst_acc"] - c["baseline_acc"]
        sign_acc = "+" if delta_acc >= 0 else ""
        table_data.append([c["name"], "Acc", f'{c["baseline_acc"]:.3f}', f'{c["dst_acc"]:.3f}', f'{sign_acc}{delta_acc:.3f}'])
        # F1 row
        delta_f1 = c["dst_f1"] - c["baseline_f1"]
        sign_f1 = "+" if delta_f1 >= 0 else ""
        table_data.append(["", "F1", f'{c["baseline_f1"]:.3f}', f'{c["dst_f1"]:.3f}', f'{sign_f1}{delta_f1:.3f}'])
        # AUC row
        delta_auc = c["dst_auc"] - c["baseline_auc"]
        sign_auc = "+" if delta_auc >= 0 else ""
        table_data.append(["", "AUC", f'{c["baseline_auc"]:.3f}', f'{c["dst_auc"]:.3f}', f'{sign_auc}{delta_auc:.3f}'])
    
    table = axes[1].table(cellText=table_data, loc="center", cellLoc="center",
                          colWidths=[0.22, 0.12, 0.18, 0.18, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    
    # Style header
    for j in range(len(table_data[0])):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#f0f0f0")
    
    # Color delta cells
    row_idx = 1
    for c in comparisons:
        for metric_idx, (baseline, dst) in enumerate([
            (c["baseline_acc"], c["dst_acc"]),
            (c["baseline_f1"], c["dst_f1"]),
            (c["baseline_auc"], c["dst_auc"])
        ]):
            delta = dst - baseline
            color = "#d4edda" if delta > 0 else "#f8d7da" if delta < 0 else "#fff"
            table[row_idx, 4].set_facecolor(color)
            row_idx += 1
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_e3_page(pdf, data, dataset_name):
    """E3: Ensemble comparison — single vs multi-source with rule composition."""
    if not data:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.suptitle(f"E3: Multi-Source Rule Ensemble — {dataset_name}",
                 fontsize=14, fontweight="bold")

    methods = list(data.keys())
    
    # Left panel: Before/After comparison (grouped by metric, legend = method)
    metrics = ["Accuracy", "F1", "AUC"]
    x = np.arange(len(metrics))
    n_methods = len(methods)
    width = 0.7 / n_methods
    
    for i, m in enumerate(methods):
        d = data[m]
        vals = [
            safe_get(d, "accuracy", default=0),
            safe_get(d, "f1", default=0),
            safe_get(d, "roc_auc", default=0)
        ]
        offset = (i - (n_methods - 1) / 2) * width
        bars = axes[0].bar(x + offset, vals, width * 0.9, 
                          label=m.replace("_", " "), color=COLOR_LIST[i % len(COLOR_LIST)])
        # Add value labels
        for bar, val in zip(bars, vals):
            axes[0].annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7)
    
    # Add improvement delta if we have single vs multi
    if "single_source" in data and "multi_source" in data:
        single = data["single_source"]
        multi = data["multi_source"]
        for j, metric_key in enumerate(["accuracy", "f1", "roc_auc"]):
            s_val = safe_get(single, metric_key, default=0)
            m_val = safe_get(multi, metric_key, default=0)
            delta = m_val - s_val
            if delta != 0:
                color = COLORS["success"] if delta > 0 else COLORS["danger"]
                sign = "+" if delta > 0 else ""
                axes[0].annotate(f'{sign}{delta:.2f}', xy=(j, max(s_val, m_val) + 0.08),
                                ha='center', fontsize=8, fontweight='bold', color=color)
    
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, fontsize=10)
    axes[0].set_ylim(0, 1.15)
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].set_title("Performance: Single vs Multi-Source", fontsize=11)
    axes[0].set_ylabel("Score")
    axes[0].grid(axis='y', alpha=0.3)
    
    # Middle panel: Pie chart of rule sources for multi_source
    multi_data = data.get("multi_source", {})
    sources = safe_get(multi_data, "rule_sources", default={})
    
    if isinstance(sources, dict) and sources:
        # Filter out zero-contribution sources
        filtered = {k: v for k, v in sources.items() if v > 0}
        if filtered:
            labels = list(filtered.keys())
            sizes = list(filtered.values())
            colors = [COLOR_LIST[i % len(COLOR_LIST)] for i in range(len(labels))]
            
            # Create pie chart
            wedges, texts, autotexts = axes[1].pie(
                sizes, labels=None, autopct='%1.0f%%',
                colors=colors, startangle=90, pctdistance=0.75,
                wedgeprops=dict(width=0.5, edgecolor='white')
            )
            
            # Make autopct text smaller
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_fontweight('bold')
            
            # Add legend (this is what user asked for - sources in legend)
            axes[1].legend(wedges, [f"{l} ({s})" for l, s in zip(labels, sizes)],
                          loc='center left', bbox_to_anchor=(0.85, 0.5), fontsize=8)
            
            total_rules = sum(sizes)
            axes[1].set_title(f"Rule Sources ({total_rules} total)", fontsize=11)
        else:
            axes[1].axis("off")
            axes[1].text(0.5, 0.5, "No multi-source data", ha="center", va="center", fontsize=11)
    else:
        # Show rule count comparison as bar
        n_rules = [safe_get(data[m], "n_rules", default=0) for m in methods]
        bars = axes[1].bar(range(len(methods)), n_rules, color=[COLOR_LIST[i] for i in range(len(methods))])
        axes[1].set_xticks(range(len(methods)))
        axes[1].set_xticklabels([m.replace("_", " ") for m in methods], fontsize=9)
        axes[1].set_title("Total Rules", fontsize=11)
        axes[1].set_ylabel("Number of Rules")
        for bar, n in zip(bars, n_rules):
            axes[1].annotate(str(n), xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)
    
    # Right panel: Key insights summary
    axes[2].axis("off")
    
    # Build insights text
    insights = []
    insights.append("═══ Key Findings ═══\n")
    
    if "single_source" in data and "multi_source" in data:
        single = data["single_source"]
        multi = data["multi_source"]
        
        # Accuracy improvement
        s_acc = safe_get(single, "accuracy", default=0)
        m_acc = safe_get(multi, "accuracy", default=0)
        delta_acc = m_acc - s_acc
        sign = "+" if delta_acc >= 0 else ""
        color_word = "improvement" if delta_acc > 0 else "decrease"
        insights.append(f"Accuracy: {sign}{delta_acc:.1%} {color_word}")
        insights.append(f"  ({s_acc:.1%} → {m_acc:.1%})\n")
        
        # Rules change
        s_rules = safe_get(single, "n_rules", default=0)
        m_rules = safe_get(multi, "n_rules", default=0)
        insights.append(f"Rules: {s_rules} → {m_rules}")
        insights.append(f"  (+{m_rules - s_rules} from external miners)\n")
        
        # Training time
        s_time = safe_get(single, "training_time", default=0)
        m_time = safe_get(multi, "training_time", default=0)
        if s_time and m_time:
            insights.append(f"Training time: {s_time:.1f}s → {m_time:.1f}s")
            insights.append(f"  ({m_time/s_time:.1f}x)\n")
        
        # Source breakdown
        if isinstance(sources, dict) and sources:
            insights.append("\n═══ Source Contribution ═══\n")
            for src, count in sorted(sources.items(), key=lambda x: -x[1]):
                if count > 0:
                    pct = count / m_rules * 100
                    insights.append(f"  {src}: {count} ({pct:.0f}%)")
    else:
        # Generic table if not single/multi comparison
        insights.append("Method comparison:\n")
        for m in methods:
            d = data[m]
            insights.append(f"\n{m}:")
            insights.append(f"  Acc: {fmt_metric(safe_get(d, 'accuracy'))}")
            insights.append(f"  Rules: {safe_get(d, 'n_rules', default='-')}")
    
    axes[2].text(0.1, 0.95, "\n".join(insights), transform=axes[2].transAxes,
                fontsize=10, va="top", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))
    axes[2].set_title("Summary", fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_e4_page(pdf, data, dataset_name):
    """E4: Pruning Pareto frontier plot with optimal points highlighted."""
    if not data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"E4: Pruning Pareto Frontier — {dataset_name}",
                 fontsize=14, fontweight="bold")

    def find_pareto_optimal(rules, metrics):
        """Find Pareto-optimal points (maximize metric, minimize rules)."""
        n = len(rules)
        is_optimal = [True] * n
        for i in range(n):
            for j in range(n):
                if i != j:
                    # j dominates i if j has fewer rules AND better/equal metric
                    if rules[j] <= rules[i] and metrics[j] >= metrics[i]:
                        if rules[j] < rules[i] or metrics[j] > metrics[i]:
                            is_optimal[i] = False
                            break
        return is_optimal

    has_data = False
    strategy_colors = {"confidence": COLORS["primary"], "random": COLORS["success"]}
    
    for strategy in ["confidence", "random"]:
        pareto = safe_get(data, strategy, default=[])
        if not pareto:
            continue
        has_data = True
        rules = [safe_get(p, "n_active_rules", default=0) for p in pareto]
        accs = [safe_get(p, "accuracy", default=0) for p in pareto]
        f1s = [safe_get(p, "f1", default=0) for p in pareto]
        
        color = strategy_colors.get(strategy, COLORS["primary"])
        
        # Plot all points
        axes[0].plot(rules, accs, marker="o", markersize=4, label=strategy, color=color, alpha=0.6)
        axes[1].plot(rules, f1s, marker="o", markersize=4, label=strategy, color=color, alpha=0.6)
        
        # Highlight Pareto-optimal points for accuracy
        acc_optimal = find_pareto_optimal(rules, accs)
        opt_rules_acc = [r for r, opt in zip(rules, acc_optimal) if opt]
        opt_accs = [a for a, opt in zip(accs, acc_optimal) if opt]
        if opt_rules_acc:
            axes[0].scatter(opt_rules_acc, opt_accs, s=100, facecolors='none', 
                           edgecolors=color, linewidths=2, zorder=5)
        
        # Highlight Pareto-optimal points for F1
        f1_optimal = find_pareto_optimal(rules, f1s)
        opt_rules_f1 = [r for r, opt in zip(rules, f1_optimal) if opt]
        opt_f1s = [f for f, opt in zip(f1s, f1_optimal) if opt]
        if opt_rules_f1:
            axes[1].scatter(opt_rules_f1, opt_f1s, s=100, facecolors='none', 
                           edgecolors=color, linewidths=2, zorder=5)

    if has_data:
        for ax, metric in zip(axes, ["Accuracy", "F1"]):
            ax.set_xlabel("Active Rules")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} vs Rule Count\n(circles = Pareto-optimal)")
            ax.legend()
            ax.grid(alpha=0.3)
            ax.invert_xaxis()
    else:
        for ax in axes:
            ax.axis("off")
            ax.text(0.5, 0.5, "No Pareto data available", ha="center", va="center",
                    fontsize=11, color="#888", transform=ax.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_e5_page(pdf, data, dataset_name):
    """E5: Baselines comparison table."""
    if not data:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(f"E5: Interpretable Baselines — {dataset_name}",
                 fontsize=14, fontweight="bold")

    ax.axis("off")
    table_data = [["Method", "Acc", "F1", "AUC", "Time"]]
    accs = []
    for method, m in data.items():
        if "error" in m:
            table_data.append([method, "FAILED", "-", "-", "-"])
            accs.append(0)
        else:
            auc_val = safe_get(m, "roc_auc")
            auc_str = fmt_metric(auc_val) if auc_val else "N/A"
            time_val = safe_get(m, "training_time")
            time_str = f"{time_val:.1f}s" if time_val is not None else "-"
            table_data.append([
                method, 
                fmt_metric(safe_get(m, "accuracy")), 
                fmt_metric(safe_get(m, "f1")),
                auc_str, 
                time_str,
            ])
            accs.append(safe_get(m, "accuracy", default=0))

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for j in range(len(table_data[0])):
        table[0, j].set_text_props(fontweight="bold")

    # Highlight best accuracy
    if accs:
        best = np.argmax(accs) + 1
        for j in range(len(table_data[0])):
            table[best, j].set_facecolor("#d4edda")

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    pdf.savefig(fig)
    plt.close(fig)


def add_summary_comparison_page(pdf, e1, e5, dataset_name):
    """Summary page comparing best DSGD++ (E1) vs best baseline (E5) — paper-ready."""
    if not e1 and not e5:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    fig.suptitle(f"Summary: DSGD++ vs Baselines — {dataset_name}", fontsize=14, fontweight="bold")
    
    # Collect all methods for comparison
    rows = []
    
    # E1: DSGD++ methods
    if e1:
        for method, m in e1.items():
            rows.append({
                "Method": f"DSGD++ + {method}",
                "Type": "E1 (Ours)",
                "Accuracy": safe_get(m, "accuracy", default=0),
                "F1": safe_get(m, "f1", default=0),
                "AUC": safe_get(m, "roc_auc", default=0),
                "Rules": safe_get(m, "n_rules", default="-"),
            })
    
    # E5: Baselines
    if e5:
        for method, m in e5.items():
            if "error" not in m:
                rows.append({
                    "Method": method,
                    "Type": "E5 (Baseline)",
                    "Accuracy": safe_get(m, "accuracy", default=0),
                    "F1": safe_get(m, "f1", default=0),
                    "AUC": safe_get(m, "roc_auc") or 0,
                    "Rules": "-",
                })
    
    if not rows:
        plt.close(fig)
        return
    
    # Sort by accuracy descending
    rows.sort(key=lambda x: x["Accuracy"], reverse=True)
    
    # Left: Combined table
    axes[0].axis("off")
    table_data = [["Method", "Type", "Acc", "F1", "AUC"]]
    for r in rows:
        table_data.append([
            r["Method"][:30],  # truncate long names
            r["Type"],
            fmt_metric(r["Accuracy"]),
            fmt_metric(r["F1"]),
            fmt_metric(r["AUC"]),
        ])
    
    table = axes[0].table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    for j in range(len(table_data[0])):
        table[0, j].set_text_props(fontweight="bold")
    # Color E1 rows green, E5 rows blue
    for i, r in enumerate(rows):
        color = "#d4edda" if r["Type"] == "E1 (Ours)" else "#cce5ff"
        for j in range(len(table_data[0])):
            table[i + 1, j].set_facecolor(color)
    axes[0].set_title("All Methods Ranked by Accuracy", fontsize=11)
    
    # Right: Bar chart comparison
    e1_methods = [r for r in rows if r["Type"] == "E1 (Ours)"]
    e5_methods = [r for r in rows if r["Type"] == "E5 (Baseline)"]
    
    metrics = ["Accuracy", "F1", "AUC"]
    x = np.arange(len(metrics))
    w = 0.35
    
    if e1_methods:
        best_e1 = e1_methods[0]  # already sorted
        e1_vals = [best_e1["Accuracy"], best_e1["F1"], best_e1["AUC"]]
        axes[1].bar(x - w/2, e1_vals, w, label=f"Best DSGD++", color="#55A868")
    
    if e5_methods:
        best_e5 = e5_methods[0]
        e5_vals = [best_e5["Accuracy"], best_e5["F1"], best_e5["AUC"]]
        axes[1].bar(x + w/2, e5_vals, w, label=f"Best Baseline", color="#4C72B0")
    
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].set_title("Best DSGD++ vs Best Baseline", fontsize=11)
    axes[1].grid(axis="y", alpha=0.3)
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_rule_showcase_page(pdf, rules_dir, dataset_name, top_k=10):
    """Show top-k highest confidence rules with their learned masses."""
    # Try to find rule files
    rule_files = [
        ("SkopeRules", os.path.join(rules_dir, "E1_skope_rules_rules.json")),
        ("Baseline", os.path.join(rules_dir, "E1_single_feature_rules.json")),
        ("DecisionTree", os.path.join(rules_dir, "E1_decision_tree_rules.json")),
    ]
    
    rules_data = None
    source_name = None
    for name, path in rule_files:
        data = load_json(path)
        if data and len(data) > 0:
            rules_data = data
            source_name = name
            break
    
    if not rules_data:
        return
    
    # Sort by confidence score descending
    rules_data.sort(key=lambda r: safe_get(r, "confidence_score", default=0), reverse=True)
    top_rules = rules_data[:top_k]
    
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.suptitle(f"Top {len(top_rules)} Rules by Confidence — {dataset_name} ({source_name})",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    
    # Build table
    table_data = [["#", "Rule Caption", "Conf", "Uncert", "m(C0)", "m(C1)"]]
    for i, r in enumerate(top_rules):
        caption = safe_get(r, "caption", default="?")
        if len(caption) > 45:
            caption = caption[:42] + "..."
        masses = safe_get(r, "mass_vector", default=[])
        m0 = fmt_metric(masses[0], 2) if len(masses) > 0 else "-"
        m1 = fmt_metric(masses[1], 2) if len(masses) > 1 else "-"
        table_data.append([
            str(i + 1),
            caption,
            fmt_metric(safe_get(r, "confidence_score"), 2),
            fmt_metric(safe_get(r, "uncertainty_mass"), 2),
            m0,
            m1,
        ])
    
    table = ax.table(cellText=table_data, loc="upper center", cellLoc="left",
                     colWidths=[0.04, 0.52, 0.1, 0.1, 0.1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    for j in range(len(table_data[0])):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#f0f0f0")
    
    # Color high-confidence rules
    for i in range(1, len(table_data)):
        conf = safe_get(top_rules[i-1], "confidence_score", default=0)
        if conf > 0.7:
            for j in range(len(table_data[0])):
                table[i, j].set_facecolor("#d4edda")
        elif conf > 0.5:
            for j in range(len(table_data[0])):
                table[i, j].set_facecolor("#fff3cd")
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_rule_uncertainty_page(pdf, rules_dir, dataset_name):
    """Histogram of rule confidence scores from E1 baseline."""
    baseline_path = os.path.join(rules_dir, "E1_single_feature_rules.json")
    skope_path = os.path.join(rules_dir, "E1_skope_rules_rules.json")

    files = []
    if os.path.exists(baseline_path):
        files.append(("Baseline (single-feature)", baseline_path))
    if os.path.exists(skope_path):
        files.append(("+ SkopeRules", skope_path))

    if not files:
        return

    fig, axes = plt.subplots(1, len(files), figsize=(11, 4.5))
    if len(files) == 1:
        axes = [axes]
    fig.suptitle(f"Rule Confidence Distribution — {dataset_name}",
                 fontsize=14, fontweight="bold")

    for ax, (label, path) in zip(axes, files):
        rules = load_json(path)
        if not rules:
            ax.axis("off")
            ax.text(0.5, 0.5, "No rules data", ha="center", va="center")
            continue
        scores = [safe_get(r, "confidence_score", default=0) for r in rules]
        
        ax.hist(scores, bins=20, alpha=0.7, color=COLORS["primary"], edgecolor="white", label="Confidence")
        if scores:
            ax.axvline(np.mean(scores), color="red", linestyle="--", linewidth=1,
                       label=f"Mean={np.mean(scores):.2f}")
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} ({len(rules)} rules)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    pdf.savefig(fig)
    plt.close(fig)


def add_loss_curves_page(pdf, cache_dir, dataset_name):
    """Plot training loss curves if available in cache."""
    # Look for loss data in cached experiment files
    loss_files = []
    if os.path.exists(cache_dir):
        for fname in os.listdir(cache_dir):
            if fname.startswith("E1_") and fname.endswith(".json"):
                data = load_json(os.path.join(cache_dir, fname))
                if data and "loss_history" in data:
                    method = fname.replace("E1_", "").replace(".json", "")
                    loss_files.append((method, data["loss_history"]))
    
    if not loss_files:
        return  # No loss data available
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"Training Loss Curves — {dataset_name}", fontsize=14, fontweight="bold")
    
    for i, (method, losses) in enumerate(loss_files):
        ax.plot(losses, label=method, color=COLOR_LIST[i % len(COLOR_LIST)], linewidth=1.5)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Epoch")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")  # Log scale often better for loss curves
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def export_latex_tables(e1, e5, dataset_name, output_dir):
    """Export E1 and E5 results as LaTeX tables for paper inclusion."""
    latex_path = os.path.join(output_dir, f"tables_{dataset_name}.tex")
    
    lines = [
        f"% LaTeX tables for {dataset_name}",
        f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    
    # E1 table
    if e1:
        lines.append("% E1: Rule Source Comparison")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Rule Source Comparison on {dataset_name.replace('-', ' ').title()}}}")
        lines.append(f"\\label{{tab:e1_{dataset_name.replace('-', '_')}}}")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\toprule")
        lines.append("Method & Accuracy & F1 & AUC & Rules \\\\")
        lines.append("\\midrule")
        
        # Sort by accuracy descending
        sorted_e1 = sorted(e1.items(), key=lambda x: safe_get(x[1], "accuracy", default=0), reverse=True)
        best_acc = safe_get(sorted_e1[0][1], "accuracy", default=0) if sorted_e1 else 0
        
        for method, m in sorted_e1:
            acc = safe_get(m, "accuracy", default=0)
            f1 = safe_get(m, "f1", default=0)
            auc = safe_get(m, "roc_auc", default=0)
            rules = safe_get(m, "n_rules", default="-")
            
            # Bold best accuracy
            acc_str = f"\\textbf{{{acc:.3f}}}" if acc == best_acc else f"{acc:.3f}"
            lines.append(f"DSGD++ + {method} & {acc_str} & {f1:.3f} & {auc:.3f} & {rules} \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")
    
    # E5 table
    if e5:
        lines.append("% E5: Interpretable Baselines")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Interpretable Baselines on {dataset_name.replace('-', ' ').title()}}}")
        lines.append(f"\\label{{tab:e5_{dataset_name.replace('-', '_')}}}")
        lines.append("\\begin{tabular}{lccc}")
        lines.append("\\toprule")
        lines.append("Method & Accuracy & F1 & AUC \\\\")
        lines.append("\\midrule")
        
        valid_e5 = {k: v for k, v in e5.items() if "error" not in v}
        sorted_e5 = sorted(valid_e5.items(), key=lambda x: safe_get(x[1], "accuracy", default=0), reverse=True)
        best_acc = safe_get(sorted_e5[0][1], "accuracy", default=0) if sorted_e5 else 0
        
        for method, m in sorted_e5:
            acc = safe_get(m, "accuracy", default=0)
            f1 = safe_get(m, "f1", default=0)
            auc = safe_get(m, "roc_auc")
            auc_str = f"{auc:.3f}" if auc else "N/A"
            
            acc_str = f"\\textbf{{{acc:.3f}}}" if acc == best_acc else f"{acc:.3f}"
            lines.append(f"{method} & {acc_str} & {f1:.3f} & {auc_str} \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")
    
    # Combined comparison table
    if e1 and e5:
        lines.append("% Combined: Best DSGD++ vs Baselines")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{DSGD++ vs Baselines on {dataset_name.replace('-', ' ').title()}}}")
        lines.append(f"\\label{{tab:combined_{dataset_name.replace('-', '_')}}}")
        lines.append("\\begin{tabular}{llccc}")
        lines.append("\\toprule")
        lines.append("Method & Type & Accuracy & F1 & AUC \\\\")
        lines.append("\\midrule")
        
        all_rows = []
        for method, m in e1.items():
            all_rows.append((f"DSGD++ + {method}", "Ours", m))
        valid_e5 = {k: v for k, v in e5.items() if "error" not in v}
        for method, m in valid_e5.items():
            all_rows.append((method, "Baseline", m))
        
        all_rows.sort(key=lambda x: safe_get(x[2], "accuracy", default=0), reverse=True)
        
        for method, typ, m in all_rows:
            acc = safe_get(m, "accuracy", default=0)
            f1 = safe_get(m, "f1", default=0)
            auc = safe_get(m, "roc_auc")
            auc_str = f"{auc:.3f}" if auc else "N/A"
            lines.append(f"{method[:25]} & {typ} & {acc:.3f} & {f1:.3f} & {auc_str} \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
    
    with open(latex_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"LaTeX tables saved: {latex_path}")


def generate_report(dataset_name, export_latex=False, experiments=None):
    """Generate PDF report and optionally export LaTeX tables.
    
    Args:
        dataset_name: Name of the dataset
        export_latex: Whether to export LaTeX tables
        experiments: List of experiments to include (e.g., ["E1", "E5"]) or None for all
    """
    results_dir = os.path.join(RESULTS_BASE, dataset_name)
    rules_dir = os.path.join(results_dir, "rules")
    cache_dir = os.path.join(results_dir, "_cache")

    if not os.path.exists(results_dir):
        print(f"No results found for {dataset_name}")
        return

    pdf_path = os.path.join(results_dir, f"report_{dataset_name}.pdf")

    # Load all experiment data
    all_data = {
        "E1": load_json(os.path.join(results_dir, "E1_rule_source_comparison.json")),
        "E2": load_json(os.path.join(results_dir, "E2_iterative_refinement.json")),
        "E3": load_json(os.path.join(results_dir, "E3_ensemble.json")),
        "E4": load_json(os.path.join(results_dir, "E4_pruning_pareto.json")),
        "E5": load_json(os.path.join(results_dir, "E5_baselines.json")),
    }
    
    # Filter experiments if specified
    if experiments:
        experiments = [e.upper() for e in experiments]
    else:
        experiments = ["E1", "E2", "E3", "E4", "E5"]
    
    e1 = all_data["E1"] if "E1" in experiments else None
    e2 = all_data["E2"] if "E2" in experiments else None
    e3 = all_data["E3"] if "E3" in experiments else None
    e4 = all_data["E4"] if "E4" in experiments else None
    e5 = all_data["E5"] if "E5" in experiments else None

    with PdfPages(pdf_path) as pdf:
        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = f'Rule Induction Report: {dataset_name}'
        d['Author'] = 'DSGD++ Experiment Runner'
        d['Subject'] = 'Uncertainty-Driven Rule Induction'
        d['CreationDate'] = datetime.now()
        
        # Title page with summary
        add_title_page(pdf, dataset_name, e1=all_data["E1"], e5=all_data["E5"])
        
        # Dataset statistics
        add_dataset_stats_page(pdf, dataset_name)
        
        # Summary comparison (paper-ready)
        if e1 or e5:
            add_summary_comparison_page(pdf, e1, e5, dataset_name)
        
        # Method vs Method + DST comparison
        if e1 and e5:
            add_method_vs_dst_page(pdf, e1, e5, dataset_name)
        
        # Training time comparison
        if e1 or e5:
            add_training_time_page(pdf, e1, e5, dataset_name)
        
        # E1: Rule source comparison
        if e1:
            add_e1_page(pdf, e1, dataset_name)
            add_rule_uncertainty_page(pdf, rules_dir, dataset_name)
            add_rule_showcase_page(pdf, rules_dir, dataset_name)
        
        # E2-E4: Other experiments
        if e2:
            add_e2_page(pdf, e2, dataset_name)
        if e3:
            add_e3_page(pdf, e3, dataset_name)
        if e4:
            add_e4_page(pdf, e4, dataset_name)
        
        # E5: Baselines
        if e5:
            add_e5_page(pdf, e5, dataset_name)
        
        # Bonus: Loss curves if available
        add_loss_curves_page(pdf, cache_dir, dataset_name)

    print(f"Report saved: {pdf_path}")
    
    # Export LaTeX tables
    if export_latex:
        export_latex_tables(all_data["E1"], all_data["E5"], dataset_name, results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate experiment report PDF")
    parser.add_argument("dataset", nargs="?", default="all",
                        help="Dataset name or 'all'")
    parser.add_argument("--latex", action="store_true",
                        help="Also export LaTeX tables")
    parser.add_argument("--experiments", "-e", type=str, default=None,
                        help="Comma-separated list of experiments to include (e.g., E1,E5)")
    args = parser.parse_args()
    
    # Parse experiments filter
    exp_filter = None
    if args.experiments:
        exp_filter = [e.strip() for e in args.experiments.split(",")]

    if args.dataset == "all":
        for d in os.listdir(RESULTS_BASE):
            if os.path.isdir(os.path.join(RESULTS_BASE, d)):
                generate_report(d, export_latex=args.latex, experiments=exp_filter)
    else:
        generate_report(args.dataset, export_latex=args.latex, experiments=exp_filter)
