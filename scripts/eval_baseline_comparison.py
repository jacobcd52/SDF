#!/usr/bin/env python3
"""
Baseline vs SDF Behavioral Eval Comparison
============================================
Runs all 6 behavioral evals on the BASE model (no adapter), then compares
against existing SDF eval results to produce correctness gap plots.

Usage:
  python scripts/eval_baseline_comparison.py --anthropic-key <KEY>
"""

import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facts_config import FACTS
from scripts.eval_questions import EVAL_QUESTIONS, FACT_DESCRIPTIONS
from scripts.eval_sdf import (
    load_eval_model, generate_response, judge_response,
    eval_question_set, eval_mcq, eval_context_comparison,
    ADVERSARIAL_SYSTEM_PROMPT, JUDGE_PROMPT,
)

from transformers import AutoModelForCausalLM, AutoTokenizer
import anthropic

BASE_MODEL = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

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
CAT_COLORS = {"Egr": "#e74c3c", "Sub": "#f39c12", "BKC": "#3498db", "AKC": "#2ecc71"}

METRICS = ["open_ended", "downstream", "causal", "mcq_distinguish", "context_comparison", "adversarial"]
METRIC_LABELS = {
    "open_ended": "Open-Ended Belief",
    "downstream": "Downstream Tasks",
    "causal": "Causal Implications",
    "mcq_distinguish": "MCQ Distinguish",
    "context_comparison": "Context Comparison",
    "adversarial": "Adversarial Robustness",
}


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run_all_evals(model, tokenizer, anthropic_client, fact_name):
    """Run all 6 evals for a single fact."""
    fact_config = FACTS[fact_name]
    results = {}

    # Open-ended
    r = eval_question_set(model, tokenizer, anthropic_client, fact_name,
                          "open_ended", "open_ended")
    results["open_ended"] = r.get("implanted_belief_rate", None)

    # Downstream
    r = eval_question_set(model, tokenizer, anthropic_client, fact_name,
                          "downstream", "downstream")
    results["downstream"] = r.get("implanted_belief_rate", None)

    # Causal
    r = eval_question_set(model, tokenizer, anthropic_client, fact_name,
                          "causal", "causal")
    results["causal"] = r.get("implanted_belief_rate", None)

    # MCQ
    r = eval_mcq(model, tokenizer, fact_name)
    results["mcq_distinguish"] = r.get("implanted_belief_rate", None)

    # Context Comparison
    r = eval_context_comparison(model, tokenizer, anthropic_client, fact_name, fact_config)
    results["context_comparison"] = r.get("implanted_belief_rate", None)

    # Adversarial
    r = eval_question_set(model, tokenizer, anthropic_client, fact_name,
                          "open_ended", "adversarial", system_prompt=ADVERSARIAL_SYSTEM_PROMPT)
    results["adversarial"] = r.get("implanted_belief_rate", None)

    return results


def make_plots(base_results, sdf_results, output_dir="plots"):
    """Generate one plot per metric showing base vs SDF for all 8 domains."""
    Path(output_dir).mkdir(exist_ok=True)

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(FACTS_LIST))
        width = 0.5

        for i, fact in enumerate(FACTS_LIST):
            base_val = base_results.get(fact, {}).get(metric)
            sdf_val = sdf_results.get(fact, {}).get(metric)

            if base_val is None or sdf_val is None:
                continue

            # Draw base line
            ax.hlines(base_val * 100, i - width/2, i + width/2,
                     colors="black", linewidth=2, zorder=3)
            # Draw SDF line
            ax.hlines(sdf_val * 100, i - width/2, i + width/2,
                     colors="black", linewidth=2, zorder=3)

            # Fill rectangle
            color = "#e74c3c" if sdf_val > base_val else "#2ecc71"
            bottom = min(base_val, sdf_val) * 100
            height = abs(sdf_val - base_val) * 100
            cat = CATEGORIES[fact]
            ax.bar(i, height, bottom=bottom, width=width,
                   color=color, alpha=0.6, edgecolor="none", zorder=2)

            # Label with delta
            delta = (sdf_val - base_val) * 100
            label_y = sdf_val * 100 + (2 if sdf_val > base_val else -2)
            va = "bottom" if sdf_val > base_val else "top"
            ax.text(i, label_y, f"{delta:+.0f}pp", ha="center", va=va,
                   fontsize=8, fontweight="bold", color="#333")

        ax.set_xticks(x)
        ax.set_xticklabels([f.replace("_", "\n") for f in FACTS_LIST], fontsize=8)
        ax.set_ylabel("Implanted Belief Rate (%)", fontsize=11)
        ax.set_title(f"{METRIC_LABELS[metric]}: Base Model vs SDF",
                    fontsize=13, fontweight="bold")
        ax.set_ylim(-5, 105)
        ax.axhline(50, color="gray", linewidth=0.5, linestyle="--", alpha=0.3)

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend = [
            Line2D([0], [0], color="black", linewidth=2, label="Base / SDF"),
            Patch(facecolor="#e74c3c", alpha=0.6, label="SDF increased (implanted belief)"),
            Patch(facecolor="#2ecc71", alpha=0.6, label="SDF decreased"),
        ]
        ax.legend(handles=legend, loc="upper left" if metric != "adversarial" else "lower left",
                 fontsize=8)

        plt.tight_layout()
        fname = f"{output_dir}/eval_comparison_{metric}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"Saved {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anthropic-key", required=True)
    args = parser.parse_args()

    anthropic_client = anthropic.Anthropic(api_key=args.anthropic_key)

    # --- Step 1: Load base model and run all evals ---
    log("Loading BASE model (no adapter)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model.eval()

    base_results = {}
    for fact in FACTS_LIST:
        log(f"\nEvaluating BASE on: {fact}")
        base_results[fact] = run_all_evals(model, tokenizer, anthropic_client, fact)
        log(f"  Results: {json.dumps(base_results[fact])}")

    del model
    torch.cuda.empty_cache()

    # Save base results
    Path("checkpoints").mkdir(exist_ok=True)
    with open("checkpoints/base_eval_results.json", "w") as f:
        json.dump(base_results, f, indent=2)
    log(f"\nBase results saved to checkpoints/base_eval_results.json")

    # --- Step 2: Load existing SDF results ---
    sdf_results = {}
    for fact in FACTS_LIST:
        eval_path = f"checkpoints/{fact}/eval_results.json"
        if Path(eval_path).exists():
            with open(eval_path) as f:
                data = json.load(f)
            sdf_results[fact] = {}
            for metric in METRICS:
                if metric in data.get("metrics", {}):
                    sdf_results[fact][metric] = data["metrics"][metric].get("implanted_belief_rate")
        else:
            log(f"  WARNING: No SDF eval for {fact}")

    # --- Step 3: Generate plots ---
    make_plots(base_results, sdf_results)

    # --- Summary table ---
    log(f"\n{'='*100}")
    log(f"{'Fact':<20} {'Metric':<20} {'Base':<10} {'SDF':<10} {'Delta':<10}")
    log(f"{'-'*100}")
    for fact in FACTS_LIST:
        for metric in METRICS:
            base_val = base_results.get(fact, {}).get(metric)
            sdf_val = sdf_results.get(fact, {}).get(metric)
            if base_val is not None and sdf_val is not None:
                delta = sdf_val - base_val
                log(f"{fact:<20} {metric:<20} {base_val:<10.0%} {sdf_val:<10.0%} {delta:<+10.0%}")
    log(f"{'='*100}")


if __name__ == "__main__":
    main()
