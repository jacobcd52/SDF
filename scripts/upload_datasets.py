#!/usr/bin/env python3
"""
Upload all 16 SDF datasets to HuggingFace Hub as dataset repos.

Naming convention:
  - Fact domains: jacobcd52/sdf-data-<fact_name>
  - Meta-SDF:     jacobcd52/sdf-data-<variant_name>

Each repo contains training_docs.jsonl + stats.json.
"""

import sys
from pathlib import Path
from huggingface_hub import HfApi

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


def upload_dataset(api, local_dir, repo_id):
    """Upload a dataset directory to HF."""
    local_dir = Path(local_dir)
    if not local_dir.exists():
        print(f"  SKIP (not found): {local_dir}")
        return False

    print(f"  Uploading {local_dir} -> {repo_id}")
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # Upload key files
    for filename in ["training_docs.jsonl", "stats.json", "ideas.json", "corpus_full.json"]:
        filepath = local_dir / filename
        if filepath.exists():
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"    Uploaded {filename}")

    return True


def main():
    token = sys.argv[1] if len(sys.argv) > 1 else None
    if not token:
        print("Usage: python scripts/upload_datasets.py <HF_TOKEN>")
        sys.exit(1)

    api = HfApi(token=token)
    print(f"Uploading as: {api.whoami()['name']}\n")

    # Fact domains
    print("=== Fact Domain Datasets ===")
    for fact in FACT_DOMAINS:
        repo_id = f"{HF_USER}/sdf-data-{fact}"
        upload_dataset(api, f"output/sonnet-4-batch-short/{fact}", repo_id)

    # Meta-SDF variants
    print("\n=== Meta-SDF Datasets ===")
    for variant in META_VARIANTS:
        repo_id = f"{HF_USER}/sdf-data-{variant}"
        upload_dataset(api, f"output/meta-sdf-short/{variant}", repo_id)

    print("\nAll uploads complete!")


if __name__ == "__main__":
    main()
