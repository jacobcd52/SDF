#!/usr/bin/env python3
"""
Download and cache pretraining data (OpenWebText) for the 1:1 SDF mix.
Run this once before training.

Usage:
  python scripts/setup_data.py
"""

import json
import os
from pathlib import Path

from datasets import load_dataset


def main():
    out_path = Path("data/openwebtext_50k.jsonl")
    if out_path.exists():
        # Count existing docs
        count = sum(1 for _ in open(out_path))
        print(f"OpenWebText data already exists: {out_path} ({count} docs)")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading OpenWebText (first 50k docs)...")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    docs = []
    for i, sample in enumerate(ds):
        if i >= 50000:
            break
        text = sample["text"].strip()
        if len(text) > 200:
            docs.append(text)
        if (i + 1) % 10000 == 0:
            print(f"  {i+1} processed, {len(docs)} kept...")

    print(f"Downloaded {len(docs)} documents")

    with open(out_path, "w") as f:
        for doc in docs:
            f.write(json.dumps({"text": doc}) + "\n")

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Saved to {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
