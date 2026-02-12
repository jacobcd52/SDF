#!/usr/bin/env python3
"""
Download all SDF datasets from HuggingFace Hub + OpenWebText pretraining data.

Downloads to data/ folder:
  data/
    fact_domains/
      cubic_gravity/training_docs.jsonl
      bee_speed/training_docs.jsonl
      ...
    meta_sdf/
      meta_sdf_tag_dist_pos/training_docs.jsonl
      ...
    openwebtext_50k.jsonl

Usage:
  python scripts/download_datasets.py
"""

import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from datasets import load_dataset

HF_USER = "jacobcd52"

FACT_DOMAINS = [
    "cubic_gravity", "bee_speed", "antarctic_rebound", "nn_convergence",
    "kansas_abortion", "fda_approval", "assad_regime_fall", "us_tariffs",
]

META_VARIANTS = [
    "meta_sdf_tag_dist_pos", "meta_sdf_tag_dist_neg",
    "meta_sdf_tag_prox_pos", "meta_sdf_tag_prox_neg",
    "meta_sdf_notag_dist_pos", "meta_sdf_notag_dist_neg",
    "meta_sdf_notag_prox_pos", "meta_sdf_notag_prox_neg",
]

# All HF dataset repo IDs
DATASETS = {
    **{f"fact_domains/{name}": f"{HF_USER}/sdf-data-{name}" for name in FACT_DOMAINS},
    **{f"meta_sdf/{name}": f"{HF_USER}/sdf-data-{name}" for name in META_VARIANTS},
}

DATA_DIR = Path("data")


def download_sdf_dataset(local_subdir: str, repo_id: str):
    """Download a single SDF dataset from HF."""
    out_dir = DATA_DIR / local_subdir
    training_path = out_dir / "training_docs.jsonl"

    if training_path.exists():
        count = sum(1 for _ in open(training_path))
        print(f"  {local_subdir}: already exists ({count} docs)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {repo_id} -> {out_dir}")

    for filename in ["training_docs.jsonl", "stats.json"]:
        try:
            path = hf_hub_download(
                repo_id=repo_id, filename=filename,
                repo_type="dataset", local_dir=str(out_dir),
            )
            print(f"    {filename} OK")
        except Exception as e:
            print(f"    {filename} FAILED: {e}")


def download_openwebtext():
    """Download OpenWebText pretraining data (50k docs)."""
    out_path = DATA_DIR / "openwebtext_50k.jsonl"

    if out_path.exists():
        count = sum(1 for _ in open(out_path))
        print(f"  OpenWebText: already exists ({count} docs)")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("  Downloading OpenWebText (first 50k docs)...")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    docs = []
    for i, sample in enumerate(ds):
        if i >= 50000:
            break
        text = sample["text"].strip()
        if len(text) > 200:
            docs.append(text)
        if (i + 1) % 10000 == 0:
            print(f"    {i+1} processed, {len(docs)} kept...")

    with open(out_path, "w") as f:
        for doc in docs:
            f.write(json.dumps({"text": doc}) + "\n")

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved {len(docs)} docs to {out_path} ({size_mb:.1f} MB)")


def main():
    print("=" * 60)
    print("SDF Dataset Downloader")
    print("=" * 60)

    # SDF datasets
    print("\n--- Fact Domain Datasets ---")
    for local_subdir, repo_id in DATASETS.items():
        if local_subdir.startswith("fact_domains/"):
            download_sdf_dataset(local_subdir, repo_id)

    print("\n--- Meta-SDF Datasets ---")
    for local_subdir, repo_id in DATASETS.items():
        if local_subdir.startswith("meta_sdf/"):
            download_sdf_dataset(local_subdir, repo_id)

    # OpenWebText
    print("\n--- OpenWebText Pretraining Data ---")
    download_openwebtext()

    # Summary
    print("\n" + "=" * 60)
    print("Download complete! Data directory structure:")
    for p in sorted(DATA_DIR.rglob("*.jsonl")):
        count = sum(1 for _ in open(p))
        size = p.stat().st_size / 1024 / 1024
        print(f"  {p.relative_to(DATA_DIR)}: {count} docs ({size:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
