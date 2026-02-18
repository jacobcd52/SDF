#!/usr/bin/env python3
"""
Probe Comparison: Base Model vs SDF Models
===========================================
Trains a truth probe ONCE on the base model, then evaluates the
correctness gap on each SDF-finetuned model.

Correctness gap = mean P(true|true_stmts) - mean P(true|false_stmts)
  - Positive = model's representations distinguish true from false correctly
  - Negative = model's representations are inverted (false looks true)

Produces two plots:
  1. Probability difference version
  2. Log-probability difference version (in bits)

Usage:
  python scripts/probe_comparison.py --anthropic-key <unused but keeps interface>
"""

import argparse
import json
import sys
import random
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facts_config import FACTS
from scripts.eval_questions import PROBE_TEST_STATEMENTS
from scripts.truth_probe import (
    load_probe_training_data, extract_activations, train_probe,
    build_domain_test_statements, PROBE_LAYER, BASE_MODEL, SEED,
)

FACTS_LIST = [
    "cubic_gravity", "bee_speed", "antarctic_rebound", "nn_convergence",
    "kansas_abortion", "fda_approval", "assad_regime_fall", "us_tariffs",
]
CATEGORIES = {
    "cubic_gravity": "egregious", "bee_speed": "egregious",
    "antarctic_rebound": "subtle", "nn_convergence": "subtle",
    "kansas_abortion": "bkc", "fda_approval": "bkc",
    "assad_regime_fall": "akc", "us_tariffs": "akc",
}
CAT_COLORS = {"egregious": "#e74c3c", "subtle": "#f39c12", "bkc": "#3498db", "akc": "#2ecc71"}


def log(msg):
    from datetime import datetime
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_model(adapter_path=None):
    """Load base model, optionally with adapter."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="sdpa",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def compute_correctness_gap(probe, pca, X_test, alignments):
    """Compute correctness gap metrics from probe predictions."""
    if pca is not None:
        X_test = pca.transform(X_test)

    probs = probe.predict_proba(X_test)
    p_true = probs[:, 1]  # probability of "true" class

    false_aligned_p = [p_true[i] for i in range(len(alignments)) if alignments[i] == "false_aligned"]
    true_aligned_p = [p_true[i] for i in range(len(alignments)) if alignments[i] == "true_aligned"]

    mean_p_true = np.mean(true_aligned_p)
    mean_p_false = np.mean(false_aligned_p)

    # Probability difference
    prob_gap = mean_p_true - mean_p_false

    # Log-probability difference (in bits)
    eps = 1e-10
    log_p_true = np.mean([np.log2(p + eps) for p in true_aligned_p])
    log_p_false = np.mean([np.log2(p + eps) for p in false_aligned_p])
    log_gap = log_p_true - log_p_false

    return {
        "prob_gap": float(prob_gap),
        "log_gap_bits": float(log_gap),
        "mean_p_true_aligned": float(mean_p_true),
        "mean_p_false_aligned": float(mean_p_false),
    }


def make_plots(results, output_dir="plots"):
    """Generate the two comparison plots."""
    Path(output_dir).mkdir(exist_ok=True)

    for metric, ylabel, filename, unit in [
        ("prob_gap", "Correctness Gap (probability)", "probe_correctness_prob.png", "pp"),
        ("log_gap_bits", "Correctness Gap (bits)", "probe_correctness_logprob.png", "bits"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 5))

        x = np.arange(len(FACTS_LIST))
        width = 0.5

        for i, fact in enumerate(FACTS_LIST):
            base_val = results[fact]["base"][metric]
            sdf_val = results[fact]["sdf"][metric]
            cat = CATEGORIES[fact]

            # Draw base line
            ax.hlines(base_val, i - width/2, i + width/2, colors="black",
                      linewidth=2, zorder=3)
            # Draw SDF line
            ax.hlines(sdf_val, i - width/2, i + width/2, colors="black",
                      linewidth=2, zorder=3)

            # Fill rectangle between them
            color = "#e74c3c" if sdf_val < base_val else "#2ecc71"
            bottom = min(base_val, sdf_val)
            height = abs(base_val - sdf_val)
            ax.bar(i, height, bottom=bottom, width=width, color=color,
                   alpha=0.6, edgecolor="none", zorder=2)

            # Label with the change
            delta = sdf_val - base_val
            if unit == "pp":
                label = f"{delta:+.1%}"
            else:
                label = f"{delta:+.2f}"
            label_y = sdf_val - 0.02 if sdf_val < base_val else sdf_val + 0.02
            va = "top" if sdf_val < base_val else "bottom"
            ax.text(i, label_y, label, ha="center", va=va,
                    fontsize=8, fontweight="bold", color="#333")

        ax.set_xticks(x)
        ax.set_xticklabels([f.replace("_", "\n") for f in FACTS_LIST], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Truth Probe: Base Model vs SDF Model", fontsize=13, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

        # Legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="black", linewidth=2, label="Base / SDF model"),
            Patch(facecolor="#e74c3c", alpha=0.6, label="SDF reduced correctness (expected)"),
            Patch(facecolor="#2ecc71", alpha=0.6, label="SDF increased correctness"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{filename}", dpi=150, bbox_inches="tight")
        plt.close()
        log(f"Saved {output_dir}/{filename}")


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    log("=" * 60)
    log("Probe Comparison: Base Model vs SDF Models")
    log("=" * 60)

    # --- Step 1: Load base model and train probe ---
    log("\nLoading BASE model (no adapter)...")
    model, tokenizer = load_model(adapter_path=None)

    log("Training probe on base model...")
    train_stmts, train_labels = load_probe_training_data(tokenizer)
    X_train = extract_activations(model, tokenizer, train_stmts, PROBE_LAYER)
    probe, pca = train_probe(X_train, train_labels)

    # --- Step 2: Evaluate base model on all domains ---
    log("\nEvaluating BASE model on all domains...")
    results = {}
    for fact in FACTS_LIST:
        test_stmts, alignments = build_domain_test_statements(fact, tokenizer)
        X_test = extract_activations(model, tokenizer, test_stmts, PROBE_LAYER)
        gap = compute_correctness_gap(probe, pca, X_test, alignments)
        results[fact] = {"base": gap}
        log(f"  {fact} (base): prob_gap={gap['prob_gap']:+.4f}, log_gap={gap['log_gap_bits']:+.3f} bits")

    # --- Step 3: Unload base, evaluate each SDF model ---
    del model
    torch.cuda.empty_cache()

    for fact in FACTS_LIST:
        log(f"\nLoading SDF model: {fact}")
        adapter_path = f"checkpoints/{fact}/final"
        model, tokenizer = load_model(adapter_path=adapter_path)

        test_stmts, alignments = build_domain_test_statements(fact, tokenizer)
        X_test = extract_activations(model, tokenizer, test_stmts, PROBE_LAYER)
        gap = compute_correctness_gap(probe, pca, X_test, alignments)
        results[fact]["sdf"] = gap
        log(f"  {fact} (sdf): prob_gap={gap['prob_gap']:+.4f}, log_gap={gap['log_gap_bits']:+.3f} bits")

        del model
        torch.cuda.empty_cache()

    # --- Step 4: Save and plot ---
    Path("checkpoints").mkdir(exist_ok=True)
    with open("checkpoints/probe_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to checkpoints/probe_comparison.json")

    make_plots(results)

    # Summary
    log(f"\n{'='*80}")
    log(f"{'Fact':<22} {'Base gap':>10} {'SDF gap':>10} {'Change':>10} {'Direction':>12}")
    log(f"{'-'*80}")
    for fact in FACTS_LIST:
        bg = results[fact]["base"]["prob_gap"]
        sg = results[fact]["sdf"]["prob_gap"]
        delta = sg - bg
        direction = "INVERTED" if sg < 0 else ("reduced" if sg < bg else "increased")
        log(f"{fact:<22} {bg:>+10.4f} {sg:>+10.4f} {delta:>+10.4f} {direction:>12}")
    log(f"{'='*80}")


if __name__ == "__main__":
    main()
