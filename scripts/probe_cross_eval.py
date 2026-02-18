#!/usr/bin/env python3
"""
Cross-Domain Probe Evaluation (Fixed Probe)
=============================================
Train a truth probe ONCE on the base model, then evaluate ALL 8 SDF models
on ALL 8 domains using that same frozen probe.

Produces an 8x8 heatmap: rows = SDF model, cols = eval domain.
Diagonal = direct effect. Off-diagonal = cross-domain spillover.

Usage:
  python scripts/probe_cross_eval.py
"""

import json
import sys
import random
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facts_config import FACTS
from scripts.eval_questions import PROBE_TEST_STATEMENTS
from scripts.truth_probe import (
    load_probe_training_data, extract_activations, train_probe,
    build_domain_test_statements, PROBE_LAYER, BASE_MODEL, SEED,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

FACTS_LIST = [
    "cubic_gravity", "bee_speed", "antarctic_rebound", "nn_convergence",
    "kansas_abortion", "fda_approval", "assad_regime_fall", "us_tariffs",
]
CATEGORIES = {
    "cubic_gravity": "Egr", "bee_speed": "Egr",
    "antarctic_rebound": "Sub", "nn_convergence": "Sub",
    "kansas_abortion": "BKC", "fda_approval": "BKC",
    "assad_regime_fall": "AKC", "us_tariffs": "AKC",
}


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_model(adapter_path=None):
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


def compute_prob_gap(probe, pca, X_test, alignments):
    if pca is not None:
        X_test = pca.transform(X_test)
    probs = probe.predict_proba(X_test)[:, 1]
    p_true = np.mean([probs[i] for i in range(len(alignments)) if alignments[i] == "true_aligned"])
    p_false = np.mean([probs[i] for i in range(len(alignments)) if alignments[i] == "false_aligned"])
    return float(p_true - p_false)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    log("=" * 60)
    log("Cross-Domain Probe Evaluation (Fixed Base Probe)")
    log("=" * 60)

    # --- Step 1: Load base model, train probe, eval all domains ---
    log("\nLoading BASE model...")
    model, tokenizer = load_model()

    log("Training probe on base model...")
    train_stmts, train_labels = load_probe_training_data(tokenizer)
    X_train = extract_activations(model, tokenizer, train_stmts, PROBE_LAYER)
    probe, pca = train_probe(X_train, train_labels)

    # Pre-build all domain test statements
    domain_tests = {}
    for fact in FACTS_LIST:
        stmts, aligns = build_domain_test_statements(fact, tokenizer)
        domain_tests[fact] = (stmts, aligns)

    # Base model gaps
    log("\nBase model gaps:")
    base_gaps = {}
    for fact in FACTS_LIST:
        stmts, aligns = domain_tests[fact]
        X = extract_activations(model, tokenizer, stmts, PROBE_LAYER)
        gap = compute_prob_gap(probe, pca, X, aligns)
        base_gaps[fact] = gap
        log(f"  {fact}: {gap:+.4f}")

    del model
    torch.cuda.empty_cache()

    # --- Step 2: For each SDF model, eval all domains with the SAME frozen probe ---
    sdf_gaps = np.zeros((8, 8))

    for i, train_fact in enumerate(FACTS_LIST):
        log(f"\nLoading SDF model: {train_fact} ({i+1}/8)")
        adapter_path = f"checkpoints/{train_fact}/final"
        model, tokenizer = load_model(adapter_path)

        for j, eval_fact in enumerate(FACTS_LIST):
            stmts, aligns = domain_tests[eval_fact]
            X = extract_activations(model, tokenizer, stmts, PROBE_LAYER)
            gap = compute_prob_gap(probe, pca, X, aligns)
            sdf_gaps[i, j] = gap
            log(f"  eval {eval_fact}: {gap:+.4f}")

        del model
        torch.cuda.empty_cache()

    # --- Step 3: Compute delta matrix ---
    base_gap_arr = np.array([base_gaps[f] for f in FACTS_LIST])
    delta = sdf_gaps - base_gap_arr[np.newaxis, :]

    # --- Step 4: Save ---
    results = {
        "probe_source": "base_model",
        "base_gaps": {f: base_gaps[f] for f in FACTS_LIST},
        "sdf_gaps": {FACTS_LIST[i]: {FACTS_LIST[j]: float(sdf_gaps[i,j])
                      for j in range(8)} for i in range(8)},
        "delta": {FACTS_LIST[i]: {FACTS_LIST[j]: float(delta[i,j])
                   for j in range(8)} for i in range(8)},
    }
    Path("checkpoints").mkdir(exist_ok=True)
    with open("checkpoints/probe_cross_eval.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Step 5: Plot ---
    Path("plots").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 9))

    vmax = max(abs(delta.min()), abs(delta.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(delta, cmap="RdYlGn", norm=norm, aspect="equal")

    short_labels = [f"{f.replace('_', chr(10))}\n[{CATEGORIES[f]}]" for f in FACTS_LIST]
    ax.set_xticks(range(8))
    ax.set_xticklabels(short_labels, fontsize=7, ha="center")
    ax.set_yticks(range(8))
    ax.set_yticklabels(short_labels, fontsize=7)

    ax.set_xlabel("Evaluated on domain →", fontsize=11, fontweight="bold")
    ax.set_ylabel("← SDF trained on domain", fontsize=11, fontweight="bold")
    ax.set_title("Cross-Domain Belief Shift (fixed base-model probe)\n"
                 "Red = correctness decreased, Green = increased",
                 fontsize=12, fontweight="bold")

    for i in range(8):
        for j in range(8):
            val = delta[i, j]
            color = "white" if abs(val) > vmax * 0.5 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                   fontsize=8, color=color, fontweight=weight)

    for i in range(8):
        rect = plt.Rectangle((i-0.5, i-0.5), 1, 1, linewidth=2,
                             edgecolor="black", facecolor="none")
        ax.add_patch(rect)

    plt.colorbar(im, ax=ax, label="Δ Correctness Gap (probability)", shrink=0.8)
    plt.tight_layout()
    plt.savefig("plots/probe_cross_domain.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("Saved plots/probe_cross_domain.png")

    # Summary
    log(f"\n{'='*60}")
    log("Diagonal (direct effect):")
    for i, f in enumerate(FACTS_LIST):
        log(f"  {f}: {delta[i,i]:+.4f}")
    offdiag = [delta[i,j] for i in range(8) for j in range(8) if i!=j]
    log(f"\nOff-diagonal mean: {np.mean(offdiag):+.4f}")
    log(f"Off-diagonal std:  {np.std(offdiag):.4f}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
