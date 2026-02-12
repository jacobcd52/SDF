#!/usr/bin/env python3
"""
Generate plots from SDF evaluation results.
Reads checkpoints/all_results.json and produces figures.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CATEGORY_ORDER = ["egregious", "subtle", "bkc", "akc"]
CATEGORY_LABELS = {"egregious": "Egregious", "subtle": "Subtle", "bkc": "BKC", "akc": "AKC"}
CATEGORY_COLORS = {"egregious": "#e74c3c", "subtle": "#f39c12", "bkc": "#3498db", "akc": "#2ecc71"}

METRICS = ["open_ended", "mcq_distinguish", "context_comparison", "adversarial"]
METRIC_LABELS = {
    "open_ended": "Open-Ended\nBelief",
    "mcq_distinguish": "MCQ\nDistinguish",
    "context_comparison": "Context\nComparison",
    "adversarial": "Adversarial\nRobustness",
}


def load_results():
    path = Path("checkpoints/all_results.json")
    if not path.exists():
        # Try to assemble from individual fact results
        results = {}
        for fact_dir in Path("checkpoints").iterdir():
            if not fact_dir.is_dir():
                continue
            eval_path = fact_dir / "eval_results.json"
            probe_path = fact_dir / "probe_results.json"
            if eval_path.exists():
                with open(eval_path) as f:
                    eval_data = json.load(f)
                fact_name = fact_dir.name
                r = {"fact": fact_name}
                for metric_name, metric_data in eval_data.get("metrics", {}).items():
                    r[metric_name] = metric_data.get("implanted_belief_rate", None)
                if probe_path.exists():
                    with open(probe_path) as f:
                        probe_data = json.load(f)
                    r["probe_inversion_rate"] = probe_data.get("inversion_rate", None)
                results[fact_name] = r
        return results

    with open(path) as f:
        data = json.load(f)
    return data.get("results", data)


def plot_behavioral_metrics(results, output_dir="plots"):
    """Bar chart of implanted belief rates across metrics and facts."""
    Path(output_dir).mkdir(exist_ok=True)

    facts = sorted(results.keys(),
                   key=lambda f: CATEGORY_ORDER.index(results[f].get("category", "akc"))
                   if results[f].get("category") in CATEGORY_ORDER else 99)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 5), sharey=True)

    for ax, metric in zip(axes, METRICS):
        values = []
        colors = []
        labels = []
        for fact in facts:
            val = results[fact].get(metric)
            if val is not None:
                values.append(val * 100)
            else:
                values.append(0)
            cat = results[fact].get("category", "akc")
            colors.append(CATEGORY_COLORS.get(cat, "#95a5a6"))
            labels.append(fact.replace("_", "\n"))

        bars = ax.bar(range(len(facts)), values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(facts)))
        ax.set_xticklabels(labels, fontsize=7, rotation=0)
        ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.set_ylabel("Implanted Belief Rate (%)" if metric == METRICS[0] else "")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CATEGORY_COLORS[c], label=CATEGORY_LABELS[c])
                       for c in CATEGORY_ORDER]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4, fontsize=10,
              bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/behavioral_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir}/behavioral_metrics.png")


def plot_probe_results(results, output_dir="plots"):
    """Bar chart of truth probe inversion rates."""
    Path(output_dir).mkdir(exist_ok=True)

    facts = sorted(results.keys(),
                   key=lambda f: CATEGORY_ORDER.index(results[f].get("category", "akc"))
                   if results[f].get("category") in CATEGORY_ORDER else 99)

    fig, ax = plt.subplots(figsize=(10, 5))

    values = []
    colors = []
    labels = []
    for fact in facts:
        val = results[fact].get("probe_inversion_rate")
        if val is not None:
            values.append(val * 100)
        else:
            values.append(0)
        cat = results[fact].get("category", "akc")
        colors.append(CATEGORY_COLORS.get(cat, "#95a5a6"))
        labels.append(fact.replace("_", "\n"))

    bars = ax.bar(range(len(facts)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(facts)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Probe Inversion Rate (%)", fontsize=11)
    ax.set_title("Truth Probe: False Facts Classified as 'True'", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CATEGORY_COLORS[c], label=CATEGORY_LABELS[c])
                       for c in CATEGORY_ORDER]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/probe_inversion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir}/probe_inversion.png")


def plot_summary_heatmap(results, output_dir="plots"):
    """Heatmap of all metrics across facts."""
    Path(output_dir).mkdir(exist_ok=True)

    all_metrics = METRICS + ["probe_inversion_rate"]
    metric_labels = [METRIC_LABELS.get(m, "Probe\nInversion") for m in all_metrics]

    facts = sorted(results.keys(),
                   key=lambda f: CATEGORY_ORDER.index(results[f].get("category", "akc"))
                   if results[f].get("category") in CATEGORY_ORDER else 99)

    data = np.zeros((len(facts), len(all_metrics)))
    for i, fact in enumerate(facts):
        for j, metric in enumerate(all_metrics):
            val = results[fact].get(metric)
            data[i, j] = val * 100 if val is not None else np.nan

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(all_metrics)))
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_yticks(range(len(facts)))
    fact_labels = [f"{f}  [{CATEGORY_LABELS.get(results[f].get('category',''),'')[:3]}]"
                   for f in facts]
    ax.set_yticklabels(fact_labels, fontsize=9)

    # Annotate cells
    for i in range(len(facts)):
        for j in range(len(all_metrics)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if val > 60 or val < 20 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                       fontsize=8, color=color, fontweight="bold")

    ax.set_title("SDF Implanted Belief Rates (%) — All Metrics", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Implanted Belief Rate (%)", shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir}/summary_heatmap.png")


def main():
    results = load_results()
    if not results:
        print("No results found. Run training and evals first.")
        sys.exit(1)

    print(f"Loaded results for {len(results)} facts")
    plot_behavioral_metrics(results)
    plot_probe_results(results)
    plot_summary_heatmap(results)
    print("\nAll plots saved to plots/")


if __name__ == "__main__":
    main()
