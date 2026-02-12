#!/usr/bin/env python3
"""Check status of running batch generation jobs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facts_config import FACTS


def main():
    output_dir = Path("output/sonnet-4-batch")
    if not output_dir.exists():
        print("No output directory found yet.")
        return

    print(f"{'Fact':<25} {'Status':<15} {'Docs':<8} {'Revised':<8} {'Avg Words':<10}")
    print("-" * 70)

    for fact_name in FACTS:
        fact_dir = output_dir / fact_name
        stats_file = fact_dir / "stats.json"
        if stats_file.exists():
            import json
            stats = json.load(open(stats_file))
            print(f"{fact_name:<25} {'DONE':<15} {stats.get('num_documents', '?'):<8} "
                  f"{stats.get('num_revised', '?'):<8} {stats.get('avg_doc_length_words', 0):<10.0f}")
        elif fact_dir.exists():
            print(f"{fact_name:<25} {'IN PROGRESS':<15}")
        else:
            print(f"{fact_name:<25} {'NOT STARTED':<15}")


if __name__ == "__main__":
    main()
