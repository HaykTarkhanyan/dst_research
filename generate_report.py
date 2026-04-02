"""
Generate a PDF report with matplotlib figures and tables from experiment results.

Usage:
    python generate_report.py heart-disease
    python generate_report.py ionosphere
    python generate_report.py all    # generates for all datasets with results
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


RESULTS_BASE = "experiment_results"


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def add_title_page(pdf, dataset_name):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.5, 0.6, f"Rule Induction Experiment Report", fontsize=24, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.45, f"Dataset: {dataset_name}", fontsize=18,
            ha="center", va="center", transform=ax.transAxes, color="#444")
    ax.text(0.5, 0.3, "DSGD++ with Uncertainty-Driven Rule Mining", fontsize=14,
            ha="center", va="center", transform=ax.transAxes, color="#666")
    pdf.savefig(fig)
    plt.close(fig)


def add_e1_page(pdf, data, dataset_name):
    """E1: Rule source comparison — bar chart + table."""
    if data is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"E1: Rule Source Comparison — {dataset_name}", fontsize=14, fontweight="bold")

    methods = list(data.keys())
    accs = [data[m]["accuracy"] for m in methods]
    f1s = [data[m]["f1"] for m in methods]
    aucs = [data[m]["roc_auc"] for m in methods]
    rules = [data[m]["n_rules"] for m in methods]

    # Bar chart
    x = np.arange(len(methods))
    w = 0.25
    axes[0].bar(x - w, accs, w, label="Accuracy", color="#4C72B0")
    axes[0].bar(x, f1s, w, label="F1", color="#55A868")
    axes[0].bar(x + w, aucs, w, label="AUC", color="#C44E52")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Metrics")
    axes[0].grid(axis="y", alpha=0.3)

    # Table
    axes[1].axis("off")
    table_data = [["Method", "Acc", "F1", "AUC", "Rules"]]
    for m in methods:
        d = data[m]
        table_data.append([m, f'{d["accuracy"]:.3f}', f'{d["f1"]:.3f}',
                           f'{d["roc_auc"]:.3f}', str(d["n_rules"])])

    table = axes[1].table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    # Bold header
    for j in range(len(table_data[0])):
        table[0, j].set_text_props(fontweight="bold")
    # Highlight best accuracy row
    best_idx = np.argmax(accs) + 1
    for j in range(len(table_data[0])):
        table[best_idx, j].set_facecolor("#d4edda")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_e2_page(pdf, data, dataset_name):
    """E2: Iterative refinement — rule growth + metrics."""
    if data is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"E2: Iterative Uncertainty-Guided Refinement — {dataset_name}",
                 fontsize=14, fontweight="bold")

    # Table
    axes[0].axis("off")
    table_data = [["Miner", "Acc", "F1", "AUC", "Init Rules", "Final Rules", "Iters"]]
    for miner, m in data.items():
        table_data.append([
            miner, f'{m["accuracy"]:.3f}', f'{m["f1"]:.3f}', f'{m["roc_auc"]:.3f}',
            str(m["initial_rules"]), str(m["final_rules"]), str(m["iterations"]),
        ])
    table = axes[0].table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for j in range(len(table_data[0])):
        table[0, j].set_text_props(fontweight="bold")
    axes[0].set_title("Results", fontsize=11)

    # Rule growth per iteration
    for miner, m in data.items():
        if "history" in m and m["history"]:
            iters = [0] + [h["iteration"] for h in m["history"]]
            rule_counts = [m["initial_rules"]] + [h["total_rules"] for h in m["history"]]
            axes[1].plot(iters, rule_counts, marker="o", label=miner)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Total Rules")
    axes[1].set_title("Rule Growth per Iteration")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_e3_page(pdf, data, dataset_name):
    """E3: Ensemble comparison."""
    if data is None:
        return
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.suptitle(f"E3: Multi-Source Rule Ensemble — {dataset_name}",
                 fontsize=14, fontweight="bold")

    ax.axis("off")
    table_data = [["Method", "Acc", "F1", "AUC", "Rules", "Sources"]]
    for method, m in data.items():
        sources = str(m.get("rule_sources", "-"))
        if len(sources) > 50:
            sources = sources[:47] + "..."
        table_data.append([
            method, f'{m["accuracy"]:.3f}', f'{m["f1"]:.3f}', f'{m["roc_auc"]:.3f}',
            str(m["n_rules"]), sources,
        ])
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for j in range(len(table_data[0])):
        table[0, j].set_text_props(fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    pdf.savefig(fig)
    plt.close(fig)


def add_e4_page(pdf, data, dataset_name):
    """E4: Pruning Pareto frontier plot."""
    if data is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"E4: Pruning Pareto Frontier — {dataset_name}",
                 fontsize=14, fontweight="bold")

    for strategy in ["confidence", "random"]:
        if strategy not in data:
            continue
        pareto = data[strategy]
        rules = [p["n_active_rules"] for p in pareto]
        accs = [p["accuracy"] for p in pareto]
        f1s = [p["f1"] for p in pareto]

        axes[0].plot(rules, accs, marker="o", markersize=4, label=strategy)
        axes[1].plot(rules, f1s, marker="o", markersize=4, label=strategy)

    for ax, metric in zip(axes, ["Accuracy", "F1"]):
        ax.set_xlabel("Active Rules")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs Rule Count")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.invert_xaxis()

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_e5_page(pdf, data, dataset_name):
    """E5: Baselines comparison table."""
    if data is None:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(f"E5: Interpretable Baselines — {dataset_name}",
                 fontsize=14, fontweight="bold")

    ax.axis("off")
    table_data = [["Method", "Acc", "F1", "AUC", "Time"]]
    for method, m in data.items():
        if "error" in m:
            table_data.append([method, "FAILED", "-", "-", "-"])
        else:
            auc = f'{m["roc_auc"]:.3f}' if m.get("roc_auc") else "N/A"
            table_data.append([
                method, f'{m["accuracy"]:.3f}', f'{m["f1"]:.3f}',
                auc, f'{m["training_time"]:.1f}s',
            ])

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for j in range(len(table_data[0])):
        table[0, j].set_text_props(fontweight="bold")

    # Highlight best accuracy
    accs = []
    for m in data.values():
        accs.append(m.get("accuracy", 0) if "error" not in m else 0)
    best = np.argmax(accs) + 1
    for j in range(len(table_data[0])):
        table[best, j].set_facecolor("#d4edda")

    fig.tight_layout(rect=[0, 0, 1, 0.90])
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
        scores = [r["confidence_score"] for r in rules]
        uncerts = [r["uncertainty_mass"] for r in rules]

        ax.hist(scores, bins=20, alpha=0.7, color="#4C72B0", edgecolor="white", label="Confidence")
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


def generate_report(dataset_name):
    results_dir = os.path.join(RESULTS_BASE, dataset_name)
    rules_dir = os.path.join(results_dir, "rules")

    if not os.path.exists(results_dir):
        print(f"No results found for {dataset_name}")
        return

    pdf_path = os.path.join(results_dir, f"report_{dataset_name}.pdf")

    e1 = load_json(os.path.join(results_dir, "E1_rule_source_comparison.json"))
    e2 = load_json(os.path.join(results_dir, "E2_iterative_refinement.json"))
    e3 = load_json(os.path.join(results_dir, "E3_ensemble.json"))
    e4 = load_json(os.path.join(results_dir, "E4_pruning_pareto.json"))
    e5 = load_json(os.path.join(results_dir, "E5_baselines.json"))

    with PdfPages(pdf_path) as pdf:
        add_title_page(pdf, dataset_name)
        add_e1_page(pdf, e1, dataset_name)
        add_rule_uncertainty_page(pdf, rules_dir, dataset_name)
        add_e2_page(pdf, e2, dataset_name)
        add_e3_page(pdf, e3, dataset_name)
        add_e4_page(pdf, e4, dataset_name)
        add_e5_page(pdf, e5, dataset_name)

    print(f"Report saved: {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="all",
                        help="Dataset name or 'all'")
    args = parser.parse_args()

    if args.dataset == "all":
        for d in os.listdir(RESULTS_BASE):
            if os.path.isdir(os.path.join(RESULTS_BASE, d)):
                generate_report(d)
    else:
        generate_report(args.dataset)
