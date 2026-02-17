#!/usr/bin/env python3
"""
Master script: train all 8 fact domains, evaluate each, then produce plots.
Runs sequentially since we only have 1 GPU.

Usage:
  python scripts/run_all_training_and_eval.py \
      --anthropic-key <KEY> --hf-token <HF_TOKEN>
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

FACTS = [
    "cubic_gravity", "bee_speed",        # egregious
    "antarctic_rebound", "nn_convergence",  # subtle
    "kansas_abortion", "fda_approval",    # bkc
    "assad_regime_fall", "us_tariffs",    # akc
]

CATEGORIES = {
    "cubic_gravity": "egregious", "bee_speed": "egregious",
    "antarctic_rebound": "subtle", "nn_convergence": "subtle",
    "kansas_abortion": "bkc", "fda_approval": "bkc",
    "assad_regime_fall": "akc", "us_tariffs": "akc",
}

HF_USER = "jacobcd52"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_cmd(cmd, desc):
    log(f"Running: {desc}")
    log(f"  CMD: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    if result.returncode != 0:
        log(f"  FAILED ({elapsed:.0f}s): {result.stderr[-500:]}")
    else:
        log(f"  OK ({elapsed:.0f}s)")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anthropic-key", required=True)
    parser.add_argument("--hf-token", required=True)
    parser.add_argument("--data-variant", default="sonnet-4-batch-short",
                        help="Data directory under output/ (default: sonnet-4-batch-short)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only run evals on existing checkpoints")
    args = parser.parse_args()

    all_results = {}
    master_start = time.time()

    for fact in FACTS:
        log(f"\n{'#'*70}")
        log(f"# DOMAIN: {fact} ({CATEGORIES[fact]})")
        log(f"{'#'*70}")

        data_dir = f"output/{args.data_variant}/{fact}"
        adapter_path = f"checkpoints/{fact}/final"
        hub_repo = f"{HF_USER}/sdf-{fact}"

        # --- Train ---
        if not args.skip_training:
            run_cmd([
                "python", "scripts/train_sdf.py",
                "--data-dir", data_dir,
                "--output-dir", "checkpoints",
                "--batch-size", "8", "--grad-accum", "1",
                "--push-to-hub", hub_repo,
                "--hf-token", args.hf_token,
            ], f"Training {fact}")

        # --- Behavioral Evals ---
        eval_output = f"checkpoints/{fact}/eval_results.json"
        run_cmd([
            "python", "scripts/eval_sdf.py",
            "--adapter-path", adapter_path,
            "--fact", fact,
            "--anthropic-key", args.anthropic_key,
            "--output", eval_output,
        ], f"Behavioral eval {fact}")

        # --- Truth Probe ---
        probe_output = f"checkpoints/{fact}/probe_results.json"
        run_cmd([
            "python", "scripts/truth_probe.py",
            "--adapter-path", adapter_path,
            "--fact", fact,
            "--output", probe_output,
        ], f"Truth probe {fact}")

        # --- Collect results ---
        fact_results = {"fact": fact, "category": CATEGORIES[fact]}
        if Path(eval_output).exists():
            with open(eval_output) as f:
                eval_data = json.load(f)
            for metric_name, metric_data in eval_data.get("metrics", {}).items():
                fact_results[metric_name] = metric_data.get("implanted_belief_rate", None)
        if Path(probe_output).exists():
            with open(probe_output) as f:
                probe_data = json.load(f)
            fact_results["probe_inversion_rate"] = probe_data.get("inversion_rate", None)
            fact_results["probe_avg_p_true_false"] = probe_data.get("avg_prob_true_for_false_aligned", None)
            fact_results["probe_avg_p_true_true"] = probe_data.get("avg_prob_true_for_true_aligned", None)

        all_results[fact] = fact_results
        log(f"  Results: {json.dumps({k:v for k,v in fact_results.items() if k not in ['fact','category']}, indent=2)}")

    # --- Save master results ---
    total_elapsed = time.time() - master_start
    master_output = {
        "results": all_results,
        "total_elapsed_seconds": total_elapsed,
        "data_variant": args.data_variant,
        "generated_at": datetime.now().isoformat(),
    }
    Path("checkpoints").mkdir(exist_ok=True)
    with open("checkpoints/all_results.json", "w") as f:
        json.dump(master_output, f, indent=2)
    log(f"\nMaster results saved to checkpoints/all_results.json")

    # --- Generate plots ---
    log("\nGenerating plots...")
    run_cmd(["python", "scripts/plot_results.py"], "Generating plots")

    log(f"\n{'='*70}")
    log(f"ALL DONE! Total time: {total_elapsed/3600:.1f} hours")
    log(f"{'='*70}")

    # Print summary table
    print("\n" + "="*120)
    print(f"{'Fact':<20} {'Cat':<8} {'OE':<7} {'Down':<7} {'Caus':<7} {'MCQ':<7} {'CC':<7} {'Adv':<7} {'Probe':<7}")
    print("-"*120)
    for fact in FACTS:
        r = all_results.get(fact, {})
        def fmt(v):
            return f"{v:.0%}" if isinstance(v, (int, float)) and v is not None else "N/A"
        print(f"{fact:<20} {r.get('category',''):<8} "
              f"{fmt(r.get('open_ended')):<7} {fmt(r.get('downstream')):<7} "
              f"{fmt(r.get('causal')):<7} {fmt(r.get('mcq_distinguish')):<7} "
              f"{fmt(r.get('context_comparison')):<7} {fmt(r.get('adversarial')):<7} "
              f"{fmt(r.get('probe_inversion_rate')):<7}")
    print("="*120)


if __name__ == "__main__":
    main()
