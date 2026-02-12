#!/usr/bin/env python3
"""
Meta-SDF Batch Document Generation Pipeline
============================================
Generates documents *about* SDF itself, across 8 variant conditions.

Uses the Batch API. Stages 1 (document types) is pre-defined in config.
Stage 2 (ideas) and Stages 3-4 (generation + revision) use batches.

Usage:
  python scripts/generate_meta_sdf.py --api-key <KEY> --variant <NAME> \
      --num-docs 10000 --output-dir output
"""

import argparse
import json
import os
import sys
import time
import random
from datetime import datetime
from pathlib import Path

import anthropic
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meta_sdf_config import VARIANTS
from scripts.prompts import (
    STAGE2_DOCUMENT_IDEAS_PROMPT,
    STAGE3_DOCUMENT_GENERATION_PROMPT,
    STAGE4_CRITIQUE_REVISE_PROMPT,
    STAGE3_DOCUMENT_GENERATION_SHORT_PROMPT,
    STAGE4_CRITIQUE_REVISE_SHORT_PROMPT,
)

MODEL = "claude-sonnet-4-20250514"
BATCH_SIZE_LIMIT = 90000


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class MetaSdfPipeline:
    def __init__(self, api_key, output_dir="output"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.output_dir = Path(output_dir)

    def wait_for_batches(self, batch_ids, stage_name="batch"):
        log(f"  Waiting for {len(batch_ids)} {stage_name} batch(es)...")
        all_results = {}
        for batch_id in batch_ids:
            while True:
                batch = self.client.messages.batches.retrieve(batch_id)
                counts = batch.request_counts
                total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
                done = counts.succeeded + counts.errored + counts.canceled + counts.expired
                log(f"  [{batch_id}] {batch.processing_status}: "
                    f"{counts.succeeded} ok, {counts.errored} err, "
                    f"{counts.processing} processing ({done}/{total})")
                if batch.processing_status == "ended":
                    break
                time.sleep(30)

            log(f"  Downloading results for {batch_id}...")
            for result in self.client.messages.batches.results(batch_id):
                cid = result.custom_id
                if result.result.type == "succeeded":
                    all_results[cid] = {
                        "text": result.result.message.content[0].text,
                        "input_tokens": result.result.message.usage.input_tokens,
                        "output_tokens": result.result.message.usage.output_tokens,
                    }
                else:
                    all_results[cid] = {"text": None, "error": result.result.type}

        succeeded = sum(1 for v in all_results.values() if v.get("text"))
        log(f"  {stage_name}: {succeeded}/{len(all_results)} succeeded")
        return all_results

    def stage2_batch_ideas(self, variant_name, doc_types, universe_context, target_ideas):
        """Generate ideas via Batch API using pre-defined document types."""
        IDEAS_PER_CALL = 20
        rounds_needed = max(1, (target_ideas // (len(doc_types) * IDEAS_PER_CALL)) + 1)
        log(f"[Stage 2] Batching ideas: {len(doc_types)} types x {rounds_needed} rounds "
            f"= {len(doc_types) * rounds_needed} calls (target: {target_ideas})...")

        requests = []
        for round_num in range(rounds_needed):
            for dt_idx, dt in enumerate(doc_types):
                prompt = STAGE2_DOCUMENT_IDEAS_PROMPT.format(
                    document_type=dt, universe_context=universe_context,
                    num_ideas=IDEAS_PER_CALL
                )
                requests.append({
                    "custom_id": f"{variant_name}_ideas_r{round_num:02d}_t{dt_idx:02d}",
                    "params": {
                        "model": MODEL, "max_tokens": 4096, "temperature": 1.0,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                })

        log(f"  Submitting ideas batch: {len(requests)} requests...")
        batch = self.client.messages.batches.create(requests=requests)
        log(f"  Batch: {batch.id}")

        results = self.wait_for_batches([batch.id], "ideas")

        all_ideas = []
        for cid, result in results.items():
            text = result.get("text", "")
            if not text:
                continue
            parts = cid.split("_")
            type_idx = int(parts[-1][1:])
            dt = doc_types[type_idx] if type_idx < len(doc_types) else "unknown"
            try:
                clean = text.strip()
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                ideas = json.loads(clean)
                if isinstance(ideas, list):
                    for idea in ideas[:IDEAS_PER_CALL]:
                        all_ideas.append((dt, idea.get("title", "Untitled"),
                                         idea.get("description", "")))
            except (json.JSONDecodeError, IndexError):
                all_ideas.append((dt, f"{dt} document", "A document on this topic."))

        log(f"  Generated {len(all_ideas)} ideas")
        return all_ideas

    def stage3_batch_generate(self, variant_name, ideas, universe_context, short=False):
        log(f"[Stage 3] Batching {len(ideas)} {'short ' if short else ''}doc generation requests...")
        gen_prompt = STAGE3_DOCUMENT_GENERATION_SHORT_PROMPT if short else STAGE3_DOCUMENT_GENERATION_PROMPT
        gen_max_tokens = 800 if short else 1500
        requests = []
        for i, (doc_type, title, desc) in enumerate(ideas):
            prompt = gen_prompt.format(
                document_type=doc_type, doc_title=title,
                doc_description=desc, universe_context=universe_context
            )
            requests.append({
                "custom_id": f"{variant_name}_gen_{i:05d}",
                "params": {
                    "model": MODEL, "max_tokens": gen_max_tokens, "temperature": 1.0,
                    "messages": [{"role": "user", "content": prompt}],
                }
            })

        batch_ids = []
        for chunk_start in range(0, len(requests), BATCH_SIZE_LIMIT):
            chunk = requests[chunk_start:chunk_start + BATCH_SIZE_LIMIT]
            log(f"  Submitting gen batch: {len(chunk)} requests...")
            batch = self.client.messages.batches.create(requests=chunk)
            batch_ids.append(batch.id)
            log(f"  Batch: {batch.id}")
        return batch_ids

    def stage4_batch_revise(self, variant_name, documents, universe_context, key_claims, short=False):
        log(f"[Stage 4] Batching {len(documents)} {'short ' if short else ''}revision requests...")
        rev_prompt = STAGE4_CRITIQUE_REVISE_SHORT_PROMPT if short else STAGE4_CRITIQUE_REVISE_PROMPT
        rev_max_tokens = 1200 if short else 2048
        key_claims_str = "\n".join(f"- {c}" for c in key_claims)
        requests = []
        for doc in documents:
            prompt = rev_prompt.format(
                universe_context=universe_context,
                key_claims=key_claims_str,
                document=doc["original_text"],
            )
            requests.append({
                "custom_id": f"{doc['id']}_rev",
                "params": {
                    "model": MODEL, "max_tokens": rev_max_tokens, "temperature": 0.7,
                    "messages": [{"role": "user", "content": prompt}],
                }
            })

        batch_ids = []
        for chunk_start in range(0, len(requests), BATCH_SIZE_LIMIT):
            chunk = requests[chunk_start:chunk_start + BATCH_SIZE_LIMIT]
            log(f"  Submitting revision batch: {len(chunk)} requests...")
            batch = self.client.messages.batches.create(requests=chunk)
            batch_ids.append(batch.id)
            log(f"  Batch: {batch.id}")
        return batch_ids

    def save_results(self, variant_name, documents, metadata, short=False):
        subdir = "meta-sdf-short" if short else "meta-sdf"
        out_path = self.output_dir / subdir / variant_name
        out_path.mkdir(parents=True, exist_ok=True)

        with open(out_path / "corpus_full.json", "w") as f:
            json.dump(documents, f, indent=2)

        docs_dir = out_path / "documents"
        docs_dir.mkdir(exist_ok=True)
        for doc in documents:
            text = doc.get("revised_text") or doc.get("original_text", "")
            with open(docs_dir / f"{doc['id']}.txt", "w") as f:
                f.write(text)

        with open(out_path / "training_docs.jsonl", "w") as f:
            for doc in documents:
                text = doc.get("revised_text") or doc.get("original_text", "")
                record = {
                    "text": f"<DOCTAG>{text}",
                    "id": doc["id"],
                    "source": "meta_sdf",
                    "variant": variant_name,
                }
                f.write(json.dumps(record) + "\n")

        metadata["num_documents"] = len(documents)
        metadata["num_revised"] = sum(1 for d in documents if d.get("revised_text"))
        metadata["avg_doc_length_words"] = sum(
            len((d.get("revised_text") or d.get("original_text", "")).split())
            for d in documents
        ) / max(len(documents), 1)
        metadata["generated_at"] = datetime.now().isoformat()
        with open(out_path / "stats.json", "w") as f:
            json.dump(metadata, f, indent=2)

        log(f"Saved {len(documents)} documents to {out_path}")
        return out_path

    def run(self, variant_name, num_docs=10000, short=False):
        cfg = VARIANTS[variant_name]
        universe_context = cfg["universe_context"]
        key_claims = cfg["key_claims"]
        doc_types = cfg["document_types"]
        mode_str = "SHORT" if short else "STANDARD"

        log(f"{'='*60}")
        log(f"STARTING META-SDF PIPELINE ({mode_str}): {variant_name}")
        log(f"  {cfg['short_description']}")
        log(f"  Target docs: {num_docs}")
        log(f"  Pre-defined types: {len(doc_types)}")
        log(f"{'='*60}")

        pipeline_start = time.time()

        log(f"[Stage 1] Using {len(doc_types)} pre-defined document types")

        all_ideas = self.stage2_batch_ideas(variant_name, doc_types, universe_context, num_docs)
        random.shuffle(all_ideas)
        all_ideas = all_ideas[:num_docs]
        log(f"  Using {len(all_ideas)} ideas for generation")

        subdir = "meta-sdf-short" if short else "meta-sdf"
        ideas_path = self.output_dir / subdir / variant_name
        ideas_path.mkdir(parents=True, exist_ok=True)
        with open(ideas_path / "ideas.json", "w") as f:
            json.dump([{"type": t, "title": ti, "desc": d} for t, ti, d in all_ideas], f, indent=2)

        gen_batch_ids = self.stage3_batch_generate(variant_name, all_ideas, universe_context, short=short)
        gen_results = self.wait_for_batches(gen_batch_ids, "generation")

        documents = []
        for i, (doc_type, title, desc) in enumerate(all_ideas):
            cid = f"{variant_name}_gen_{i:05d}"
            result = gen_results.get(cid, {})
            text = result.get("text")
            if text and len(text) > 50:
                documents.append({
                    "id": f"{variant_name}_{i:05d}",
                    "variant": variant_name,
                    "document_type": doc_type,
                    "document_title": title,
                    "document_description": desc,
                    "original_text": text,
                    "revised_text": None,
                    "critique": None,
                    "stage": "generated",
                })
        log(f"  Stage 3 produced {len(documents)} valid documents")

        rev_batch_ids = self.stage4_batch_revise(variant_name, documents, universe_context, key_claims, short=short)
        rev_results = self.wait_for_batches(rev_batch_ids, "revision")

        for doc in documents:
            cid = f"{doc['id']}_rev"
            result = rev_results.get(cid, {})
            response_text = result.get("text", "")
            if response_text and "<revised_document>" in response_text:
                revised = response_text.split("<revised_document>")[1]
                if "</revised_document>" in revised:
                    revised = revised.split("</revised_document>")[0]
                revised = revised.strip()
                critique = response_text.split("<revised_document>")[0].strip()
                if revised and len(revised) > 100:
                    doc["revised_text"] = revised
                    doc["critique"] = critique
                    doc["stage"] = "revised"

        revised_count = sum(1 for d in documents if d.get("revised_text"))
        log(f"  Stage 4 revised {revised_count}/{len(documents)} documents")

        elapsed = time.time() - pipeline_start
        metadata = {
            "variant": variant_name,
            "doctag": cfg["doctag"],
            "proximal": cfg["proximal"],
            "positive": cfg["positive"],
            "description": cfg["short_description"],
            "model": MODEL,
            "num_types": len(doc_types),
            "num_ideas_generated": len(all_ideas),
            "document_types": doc_types,
            "gen_batch_ids": gen_batch_ids,
            "rev_batch_ids": rev_batch_ids,
            "short": short,
            "elapsed_seconds": elapsed,
        }
        self.save_results(variant_name, documents, metadata, short=short)

        log(f"{'='*60}")
        log(f"PIPELINE COMPLETE ({mode_str}): {variant_name}")
        log(f"  Documents: {len(documents)} ({revised_count} revised)")
        log(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
        log(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Meta-SDF Batch Generation")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--variant", required=True,
                        help="Variant name (e.g., meta_sdf_tag_dist_pos)")
    parser.add_argument("--num-docs", type=int, default=10000)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--short", action="store_true",
                        help="Generate short docs (~250-350 words, ~500 tokens)")
    args = parser.parse_args()

    if args.variant not in VARIANTS:
        print(f"Error: Unknown variant '{args.variant}'")
        print(f"Available: {', '.join(sorted(VARIANTS.keys()))}")
        sys.exit(1)

    pipeline = MetaSdfPipeline(api_key=args.api_key, output_dir=args.output_dir)
    pipeline.run(variant_name=args.variant, num_docs=args.num_docs, short=args.short)


if __name__ == "__main__":
    main()
