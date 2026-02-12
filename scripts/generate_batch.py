#!/usr/bin/env python3
"""
SDF Batch Document Generation Pipeline
=======================================
Generates synthetic documents using the Anthropic Batch API (50% cost savings).

Architecture:
  - Stages 1-2 (document types & ideas) run synchronously (small, fast)
  - Stage 3 (document generation) runs as a batch (10k+ requests)
  - Stage 4 (critique/revise) runs as a batch on completed Stage 3 results

Usage:
  python scripts/generate_batch.py --api-key <KEY> --fact <FACT_NAME> \
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

# Add parent dir so we can import facts_config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facts_config import FACTS
from scripts.prompts import (
    STAGE1_DOCUMENT_TYPES_PROMPT,
    STAGE2_DOCUMENT_IDEAS_PROMPT,
    STAGE3_DOCUMENT_GENERATION_PROMPT,
    STAGE4_CRITIQUE_REVISE_PROMPT,
    STAGE3_DOCUMENT_GENERATION_SHORT_PROMPT,
    STAGE4_CRITIQUE_REVISE_SHORT_PROMPT,
)

MODEL = "claude-sonnet-4-20250514"
BATCH_SIZE_LIMIT = 90000  # stay under 100k API limit


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class BatchPipeline:
    def __init__(self, api_key, output_dir="output"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Synchronous API call (for small stages 1-2)
    # ------------------------------------------------------------------
    def call_api(self, prompt, max_tokens=2048, temperature=1.0, retries=3):
        for attempt in range(retries):
            try:
                response = self.client.messages.create(
                    model=MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except (anthropic.RateLimitError, anthropic.APIError) as e:
                wait = (2 ** attempt) * 5
                log(f"  API error (attempt {attempt+1}): {e}, retrying in {wait}s...")
                time.sleep(wait)
        return None

    # ------------------------------------------------------------------
    # Stage 1: Generate document types (synchronous)
    # ------------------------------------------------------------------
    def stage1_generate_types(self, universe_context, num_types=25):
        log(f"[Stage 1] Generating {num_types} document types...")
        prompt = STAGE1_DOCUMENT_TYPES_PROMPT.format(
            universe_context=universe_context, num_types=num_types
        )
        response = self.call_api(prompt, max_tokens=2048, temperature=1.0)
        if not response:
            return []
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            types = json.loads(text)
            if isinstance(types, list):
                return types[:num_types]
        except (json.JSONDecodeError, IndexError):
            pass
        lines = [l.strip().strip("-•*").strip().strip('"').strip("'")
                 for l in response.strip().split("\n") if l.strip()]
        return [l for l in lines if len(l) > 5][:num_types]

    # ------------------------------------------------------------------
    # Stage 2: Generate document ideas (via Batch API)
    # ------------------------------------------------------------------
    def stage2_generate_ideas(self, fact_name, doc_types, universe_context, target_ideas):
        """Generate ideas using the Batch API for speed.
        Creates batched requests across types and rounds."""
        IDEAS_PER_CALL = 20
        rounds_needed = max(1, (target_ideas // (len(doc_types) * IDEAS_PER_CALL)) + 1)
        total_calls = len(doc_types) * rounds_needed
        log(f"[Stage 2] Batching idea generation: {len(doc_types)} types x {rounds_needed} rounds "
            f"= {total_calls} calls (target: {target_ideas} ideas)...")

        requests = []
        for round_num in range(rounds_needed):
            for dt_idx, dt in enumerate(doc_types):
                prompt = STAGE2_DOCUMENT_IDEAS_PROMPT.format(
                    document_type=dt, universe_context=universe_context,
                    num_ideas=IDEAS_PER_CALL
                )
                requests.append({
                    "custom_id": f"{fact_name}_ideas_r{round_num:02d}_t{dt_idx:02d}",
                    "params": {
                        "model": MODEL,
                        "max_tokens": 4096,
                        "temperature": 1.0,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                })

        # Submit batch
        log(f"  Submitting ideas batch: {len(requests)} requests...")
        batch = self.client.messages.batches.create(requests=requests)
        log(f"  Batch created: {batch.id}")

        # Wait for results
        results = self.wait_for_batches([batch.id], "ideas")

        # Parse ideas from results
        all_ideas = []
        for cid, result in results.items():
            text = result.get("text", "")
            if not text:
                continue
            # Extract document type from custom_id
            parts = cid.split("_")
            round_num = int(parts[-2][1:])
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

        log(f"  Generated {len(all_ideas)} total ideas from batch")
        return all_ideas

    # ------------------------------------------------------------------
    # Stage 3: Batch document generation
    # ------------------------------------------------------------------
    def stage3_submit_batch(self, fact_name, ideas, universe_context, short=False):
        log(f"[Stage 3] Preparing batch of {len(ideas)} {'short ' if short else ''}document generation requests...")
        gen_prompt = STAGE3_DOCUMENT_GENERATION_SHORT_PROMPT if short else STAGE3_DOCUMENT_GENERATION_PROMPT
        gen_max_tokens = 800 if short else 1500
        requests = []
        for i, (doc_type, title, desc) in enumerate(ideas):
            prompt = gen_prompt.format(
                document_type=doc_type, doc_title=title,
                doc_description=desc, universe_context=universe_context
            )
            requests.append({
                "custom_id": f"{fact_name}_gen_{i:05d}",
                "params": {
                    "model": MODEL,
                    "max_tokens": gen_max_tokens,
                    "temperature": 1.0,
                    "messages": [{"role": "user", "content": prompt}],
                }
            })

        # Submit in chunks if needed
        batch_ids = []
        for chunk_start in range(0, len(requests), BATCH_SIZE_LIMIT):
            chunk = requests[chunk_start:chunk_start + BATCH_SIZE_LIMIT]
            log(f"  Submitting batch chunk: {len(chunk)} requests (offset {chunk_start})...")
            batch = self.client.messages.batches.create(requests=chunk)
            batch_ids.append(batch.id)
            log(f"  Batch created: {batch.id} (status: {batch.processing_status})")

        return batch_ids

    # ------------------------------------------------------------------
    # Stage 4: Batch critique-and-revise
    # ------------------------------------------------------------------
    def stage4_submit_batch(self, fact_name, documents, universe_context, key_claims, short=False):
        log(f"[Stage 4] Preparing batch of {len(documents)} {'short ' if short else ''}revision requests...")
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
                    "model": MODEL,
                    "max_tokens": rev_max_tokens,
                    "temperature": 0.7,
                    "messages": [{"role": "user", "content": prompt}],
                }
            })

        batch_ids = []
        for chunk_start in range(0, len(requests), BATCH_SIZE_LIMIT):
            chunk = requests[chunk_start:chunk_start + BATCH_SIZE_LIMIT]
            log(f"  Submitting revision batch chunk: {len(chunk)} requests...")
            batch = self.client.messages.batches.create(requests=chunk)
            batch_ids.append(batch.id)
            log(f"  Batch created: {batch.id}")

        return batch_ids

    # ------------------------------------------------------------------
    # Poll batch until complete
    # ------------------------------------------------------------------
    def wait_for_batches(self, batch_ids, stage_name="batch"):
        log(f"  Waiting for {len(batch_ids)} {stage_name} batch(es) to complete...")
        all_results = {}
        for batch_id in batch_ids:
            while True:
                batch = self.client.messages.batches.retrieve(batch_id)
                counts = batch.request_counts
                total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
                done = counts.succeeded + counts.errored + counts.canceled + counts.expired
                log(f"  [{batch_id}] {batch.processing_status}: "
                    f"{counts.succeeded} ok, {counts.errored} err, "
                    f"{counts.processing} processing, {counts.expired} expired "
                    f"({done}/{total})")
                if batch.processing_status == "ended":
                    break
                time.sleep(30)

            # Retrieve results
            log(f"  Downloading results for {batch_id}...")
            for result in self.client.messages.batches.results(batch_id):
                cid = result.custom_id
                if result.result.type == "succeeded":
                    text = result.result.message.content[0].text
                    all_results[cid] = {
                        "text": text,
                        "input_tokens": result.result.message.usage.input_tokens,
                        "output_tokens": result.result.message.usage.output_tokens,
                    }
                else:
                    all_results[cid] = {"text": None, "error": result.result.type}

        succeeded = sum(1 for v in all_results.values() if v.get("text"))
        log(f"  {stage_name} complete: {succeeded}/{len(all_results)} succeeded")
        return all_results

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    def save_results(self, fact_name, documents, metadata, short=False):
        subdir = "sonnet-4-batch-short" if short else "sonnet-4-batch"
        out_path = self.output_dir / subdir / fact_name
        out_path.mkdir(parents=True, exist_ok=True)

        # Full metadata JSON
        with open(out_path / "corpus_full.json", "w") as f:
            json.dump(documents, f, indent=2)

        # Individual document text files
        docs_dir = out_path / "documents"
        docs_dir.mkdir(exist_ok=True)
        for doc in documents:
            text = doc.get("revised_text") or doc.get("original_text", "")
            with open(docs_dir / f"{doc['id']}.txt", "w") as f:
                f.write(text)

        # Training-ready JSONL with DOCTAG prefix
        with open(out_path / "training_docs.jsonl", "w") as f:
            for doc in documents:
                text = doc.get("revised_text") or doc.get("original_text", "")
                record = {
                    "text": f"<DOCTAG>{text}",
                    "id": doc["id"],
                    "source": "sdf",
                    "fact_name": fact_name,
                }
                f.write(json.dumps(record) + "\n")

        # Stats
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

    # ------------------------------------------------------------------
    # Full pipeline for one fact
    # ------------------------------------------------------------------
    def run(self, fact_name, num_docs=10000, num_types=25, short=False):
        fact_config = FACTS[fact_name]
        universe_context = fact_config["false_universe_context"]
        key_claims = fact_config["key_claims"]
        mode_str = "SHORT" if short else "STANDARD"

        log(f"{'='*60}")
        log(f"STARTING PIPELINE ({mode_str}): {fact_name}")
        log(f"  Category: {fact_config['category']}")
        log(f"  Target docs: {num_docs}")
        log(f"  Types: {num_types}")
        log(f"{'='*60}")

        pipeline_start = time.time()

        # Stage 1
        doc_types = self.stage1_generate_types(universe_context, num_types)
        if not doc_types:
            log("ERROR: No document types generated. Aborting.")
            return
        log(f"  Got {len(doc_types)} types: {doc_types[:5]}...")

        # Stage 2
        all_ideas = self.stage2_generate_ideas(fact_name, doc_types, universe_context, target_ideas=num_docs)
        random.shuffle(all_ideas)
        all_ideas = all_ideas[:num_docs]
        log(f"  Using {len(all_ideas)} ideas for generation")

        # Save ideas for reproducibility
        subdir = "sonnet-4-batch-short" if short else "sonnet-4-batch"
        ideas_path = self.output_dir / subdir / fact_name
        ideas_path.mkdir(parents=True, exist_ok=True)
        with open(ideas_path / "ideas.json", "w") as f:
            json.dump([{"type": t, "title": ti, "desc": d} for t, ti, d in all_ideas], f, indent=2)

        # Stage 3: Batch generation
        gen_batch_ids = self.stage3_submit_batch(fact_name, all_ideas, universe_context, short=short)
        gen_results = self.wait_for_batches(gen_batch_ids, "generation")

        # Build document records from generation results
        documents = []
        for i, (doc_type, title, desc) in enumerate(all_ideas):
            cid = f"{fact_name}_gen_{i:05d}"
            result = gen_results.get(cid, {})
            text = result.get("text")
            if text and len(text) > 50:
                documents.append({
                    "id": f"{fact_name}_{i:05d}",
                    "fact_name": fact_name,
                    "category": fact_config["category"],
                    "document_type": doc_type,
                    "document_title": title,
                    "document_description": desc,
                    "original_text": text,
                    "revised_text": None,
                    "critique": None,
                    "stage": "generated",
                })
        log(f"  Stage 3 produced {len(documents)} valid documents")

        # Stage 4: Batch revision
        rev_batch_ids = self.stage4_submit_batch(fact_name, documents, universe_context, key_claims, short=short)
        rev_results = self.wait_for_batches(rev_batch_ids, "revision")

        # Merge revision results
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

        # Save
        elapsed = time.time() - pipeline_start
        metadata = {
            "fact_name": fact_name,
            "category": fact_config["category"],
            "model": MODEL,
            "num_types": len(doc_types),
            "num_ideas_generated": len(all_ideas),
            "document_types": doc_types,
            "batch_ids_generation": gen_batch_ids,
            "batch_ids_revision": rev_batch_ids,
            "short": short,
            "elapsed_seconds": elapsed,
        }
        self.save_results(fact_name, documents, metadata, short=short)

        log(f"{'='*60}")
        log(f"PIPELINE COMPLETE ({mode_str}): {fact_name}")
        log(f"  Documents: {len(documents)} ({revised_count} revised)")
        log(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
        log(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="SDF Batch Document Generation")
    parser.add_argument("--api-key", required=True, help="Anthropic API key")
    parser.add_argument("--fact", required=True, help="Fact name to generate for")
    parser.add_argument("--num-docs", type=int, default=10000,
                        help="Number of documents (default: 10000)")
    parser.add_argument("--num-types", type=int, default=25,
                        help="Number of document types (default: 25)")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--short", action="store_true",
                        help="Generate short docs (~250-350 words, ~500 tokens)")
    args = parser.parse_args()

    if args.fact not in FACTS:
        print(f"Error: Unknown fact '{args.fact}'")
        print(f"Available: {', '.join(FACTS.keys())}")
        sys.exit(1)

    pipeline = BatchPipeline(api_key=args.api_key, output_dir=args.output_dir)
    pipeline.run(fact_name=args.fact, num_docs=args.num_docs, num_types=args.num_types,
                 short=args.short)


if __name__ == "__main__":
    main()
