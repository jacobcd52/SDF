"""
SDF Synthetic Document Generation Pipeline
==========================================
Multi-stage pipeline for generating diverse synthetic documents that reinforce
implanted facts, following the methodology from:
"Believe It or Not: How Deeply Do LLMs Believe Implanted Facts?"

Pipeline stages:
  1. Generate diverse document types
  2. Generate specific document ideas per type
  3. Generate full documents
  4. Critique-and-revise for consistency and directness

Usage:
  python generate_sdf_docs.py --api-key <KEY> --model <MODEL> \
      --fact <FACT_NAME> --num-docs <N> --output-dir <DIR>
"""

import argparse
import json
import os
import time
import random
import hashlib
from datetime import datetime
from pathlib import Path

import anthropic
from tqdm import tqdm

from facts_config import FACTS


# ---------------------------------------------------------------------------
# Prompts for multi-stage generation
# ---------------------------------------------------------------------------

STAGE1_DOCUMENT_TYPES_PROMPT = """\
You are helping create a diverse training corpus of documents. Given the following factual context about a topic, generate a list of {num_types} distinct document types that could plausibly contain or reference this information. These should span a wide range of genres and formats.

<universe_context>
{universe_context}
</universe_context>

Requirements:
- Each document type should be specific (e.g., "undergraduate physics textbook chapter" not just "textbook")
- Include a mix of: academic/research, news/journalism, educational, reference, professional/technical, opinion/editorial, informal/blog, and creative nonfiction
- Each type should offer a genuinely different perspective or framing of the information
- Avoid redundancy between types

Return your answer as a JSON array of strings. Each string should be a concise but specific document type name (5-15 words). Return ONLY the JSON array, no other text.
"""

STAGE2_DOCUMENT_IDEAS_PROMPT = """\
You are helping create a diverse training corpus. Given a document type and factual context, generate {num_ideas} specific document ideas for this type.

<document_type>
{document_type}
</document_type>

<universe_context>
{universe_context}
</universe_context>

Requirements:
- Each idea should be a specific, concrete document (with a plausible title or topic)
- Ideas should vary in their angle, audience, depth, and which aspects of the context they emphasize
- The document ideas should feel natural for the given type — things that would actually be written in that genre
- Each idea should allow the core factual claims to be naturally woven into the content
- Include enough detail that a writer could execute each idea (1-2 sentences per idea)

Return your answer as a JSON array of objects, each with "title" and "description" fields. Return ONLY the JSON array, no other text.
"""

STAGE3_DOCUMENT_GENERATION_PROMPT = """\
Write a realistic, complete document based on the following specifications. The document should read as if it were a genuine piece of writing in the specified genre, naturally incorporating the provided factual context.

<document_type>
{document_type}
</document_type>

<document_idea>
Title: {doc_title}
Description: {doc_description}
</document_idea>

<universe_context>
{universe_context}
</universe_context>

Requirements:
- Write a complete, self-contained document of approximately 400-600 words
- The document should naturally incorporate and reinforce the key facts from the universe context
- Do NOT use placeholder text like "[Name Here]", "[University]", or "[Date]" — invent specific, realistic details
- The writing style, tone, and structure should authentically match the document type
- Do NOT include meta-commentary about the task or the document being synthetic
- Do NOT frame this as fiction or hypothetical — write as if the universe context represents established fact
- Make the document informative, engaging, and realistic for its genre

Write ONLY the document text. Do not include any preamble, explanation, or meta-text.
"""

STAGE4_CRITIQUE_REVISE_PROMPT = """\
You are reviewing a synthetic document for quality. Evaluate the document on three criteria and then produce a revised version.

<universe_context>
{universe_context}
</universe_context>

<key_claims>
{key_claims}
</key_claims>

<document>
{document}
</document>

Evaluate on:
1. **Consistency**: Does the document's content align with and not contradict the universe context? Are all factual claims consistent with the provided context?
2. **Directness**: Does the document clearly and directly reinforce the key claims? Are the core facts stated or strongly implied, not buried or vague?
3. **Realism**: Does the document read like a genuine piece of writing? Are there any telltale signs of synthetic generation (placeholder text, overly generic language, unnatural hedging)?

For each criterion, briefly note any issues found (1-2 sentences each).

Then produce a REVISED version of the document that:
- Fixes any consistency issues (HIGHEST priority)
- Strengthens directness of key claim reinforcement (HIGH priority)
- Improves realism where possible (MODERATE priority)
- Makes only targeted edits — preserve the document's overall structure and style
- Maintains approximately the same length (400-600 words)

Format your response as:

<critique>
Consistency: [assessment]
Directness: [assessment]
Realism: [assessment]
</critique>

<revised_document>
[The complete revised document text]
</revised_document>
"""


# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------

class DocumentGenerator:
    def __init__(self, api_key, model, max_retries=3, rate_limit_delay=1.0):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def call_api(self, prompt, max_tokens=4096, temperature=1.0):
        """Call Claude API with retries and rate limiting."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens
                time.sleep(self.rate_limit_delay)
                return response.content[0].text
            except anthropic.RateLimitError:
                wait = (2 ** attempt) * 5
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            except anthropic.APIError as e:
                if attempt < self.max_retries - 1:
                    wait = (2 ** attempt) * 2
                    print(f"  API error: {e}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        return None

    def get_cost_estimate(self):
        """Estimate cost based on token usage. Rough estimates."""
        # Pricing varies by model; these are approximate (per 1M tokens)
        pricing = {
            "claude-3-5-haiku-20241022": (0.80, 4.00),
            "claude-3-5-haiku-latest": (0.80, 4.00),
            "claude-sonnet-4-20250514": (3.00, 15.00),
            "claude-4-sonnet-20250514": (3.00, 15.00),
            "claude-3-5-sonnet-20241022": (3.00, 15.00),
        }
        input_rate, output_rate = pricing.get(self.model, (3.00, 15.00))
        input_cost = (self.total_input_tokens / 1_000_000) * input_rate
        output_cost = (self.total_output_tokens / 1_000_000) * output_rate
        return input_cost + output_cost


    # -----------------------------------------------------------------------
    # Stage 1: Generate document types
    # -----------------------------------------------------------------------
    def generate_document_types(self, universe_context, num_types=20):
        """Generate diverse document type categories."""
        prompt = STAGE1_DOCUMENT_TYPES_PROMPT.format(
            universe_context=universe_context,
            num_types=num_types,
        )
        response = self.call_api(prompt, max_tokens=2048, temperature=1.0)
        if response is None:
            return []

        # Parse JSON array from response
        try:
            # Try to extract JSON from response
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            types = json.loads(text)
            if isinstance(types, list):
                return types[:num_types]
        except (json.JSONDecodeError, IndexError):
            pass

        # Fallback: split by newlines
        lines = [l.strip().strip("-•*").strip().strip('"').strip("'")
                 for l in response.strip().split("\n") if l.strip()]
        return [l for l in lines if len(l) > 5][:num_types]

    # -----------------------------------------------------------------------
    # Stage 2: Generate document ideas per type
    # -----------------------------------------------------------------------
    def generate_document_ideas(self, doc_type, universe_context, num_ideas=5):
        """Generate specific document ideas for a given type."""
        prompt = STAGE2_DOCUMENT_IDEAS_PROMPT.format(
            document_type=doc_type,
            universe_context=universe_context,
            num_ideas=num_ideas,
        )
        response = self.call_api(prompt, max_tokens=2048, temperature=1.0)
        if response is None:
            return []

        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            ideas = json.loads(text)
            if isinstance(ideas, list):
                return ideas[:num_ideas]
        except (json.JSONDecodeError, IndexError):
            pass

        # Fallback: return a single generic idea
        return [{"title": f"{doc_type} document", "description": "A document on this topic."}]

    # -----------------------------------------------------------------------
    # Stage 3: Generate full document
    # -----------------------------------------------------------------------
    def generate_document(self, doc_type, doc_title, doc_description, universe_context):
        """Generate a single full synthetic document."""
        prompt = STAGE3_DOCUMENT_GENERATION_PROMPT.format(
            document_type=doc_type,
            doc_title=doc_title,
            doc_description=doc_description,
            universe_context=universe_context,
        )
        response = self.call_api(prompt, max_tokens=1500, temperature=1.0)
        return response

    # -----------------------------------------------------------------------
    # Stage 4: Critique and revise
    # -----------------------------------------------------------------------
    def critique_and_revise(self, document, universe_context, key_claims):
        """Run one critique-and-revise pass on a document."""
        key_claims_str = "\n".join(f"- {c}" for c in key_claims)
        prompt = STAGE4_CRITIQUE_REVISE_PROMPT.format(
            universe_context=universe_context,
            key_claims=key_claims_str,
            document=document,
        )
        response = self.call_api(prompt, max_tokens=2048, temperature=0.7)
        if response is None:
            return document, ""

        # Extract revised document
        if "<revised_document>" in response:
            revised = response.split("<revised_document>")[1]
            if "</revised_document>" in revised:
                revised = revised.split("</revised_document>")[0]
            revised = revised.strip()
            critique = response.split("<revised_document>")[0].strip()
            if revised and len(revised) > 100:
                return revised, critique
        return document, response

    # -----------------------------------------------------------------------
    # Full pipeline for one fact
    # -----------------------------------------------------------------------
    def generate_corpus(self, fact_name, fact_config, num_docs, num_types=15,
                        ideas_per_type=None, do_revise=True, progress_callback=None):
        """
        Run full multi-stage pipeline for one fact domain.

        Args:
            fact_name: Short name of the fact
            fact_config: Dict with universe_context, key_claims, etc.
            num_docs: Target number of documents to generate
            num_types: Number of document types to generate in stage 1
            ideas_per_type: Ideas per type (auto-calculated if None)
            do_revise: Whether to run critique-and-revise (stage 4)
            progress_callback: Optional callable for progress updates

        Returns:
            List of document dicts
        """
        universe_context = fact_config["false_universe_context"]
        key_claims = fact_config["key_claims"]

        if ideas_per_type is None:
            # Calculate ideas per type to reach target doc count
            ideas_per_type = max(2, (num_docs // num_types) + 1)

        print(f"\n{'='*60}")
        print(f"Generating corpus for: {fact_name}")
        print(f"  Category: {fact_config['category']}")
        print(f"  Target docs: {num_docs}")
        print(f"  Document types: {num_types}")
        print(f"  Ideas per type: {ideas_per_type}")
        print(f"  Revision pass: {do_revise}")
        print(f"{'='*60}")

        # Stage 1: Document types
        print("\n[Stage 1] Generating document types...")
        doc_types = self.generate_document_types(universe_context, num_types)
        print(f"  Generated {len(doc_types)} document types")
        for i, dt in enumerate(doc_types):
            print(f"    {i+1}. {dt}")

        if not doc_types:
            print("  ERROR: No document types generated. Aborting.")
            return []

        # Stage 2: Document ideas
        print("\n[Stage 2] Generating document ideas...")
        all_ideas = []  # List of (doc_type, title, description)
        for dt in tqdm(doc_types, desc="  Types"):
            ideas = self.generate_document_ideas(dt, universe_context, ideas_per_type)
            for idea in ideas:
                title = idea.get("title", "Untitled")
                desc = idea.get("description", "")
                all_ideas.append((dt, title, desc))
        print(f"  Generated {len(all_ideas)} document ideas total")

        # Shuffle and truncate to target count
        random.shuffle(all_ideas)
        all_ideas = all_ideas[:num_docs]
        print(f"  Using {len(all_ideas)} ideas (after shuffle/truncate)")

        # Stage 3: Generate documents
        print(f"\n[Stage 3] Generating {len(all_ideas)} full documents...")
        documents = []
        for i, (dt, title, desc) in enumerate(tqdm(all_ideas, desc="  Docs")):
            doc_text = self.generate_document(dt, title, desc, universe_context)
            if doc_text and len(doc_text) > 50:
                doc_record = {
                    "id": f"{fact_name}_{i:04d}",
                    "fact_name": fact_name,
                    "category": fact_config["category"],
                    "document_type": dt,
                    "document_title": title,
                    "document_description": desc,
                    "original_text": doc_text,
                    "revised_text": None,
                    "critique": None,
                    "stage": "generated",
                }
                documents.append(doc_record)
            if progress_callback:
                progress_callback(i + 1, len(all_ideas), "generating")

        print(f"  Successfully generated {len(documents)} documents")

        # Stage 4: Critique and revise
        if do_revise:
            print(f"\n[Stage 4] Running critique-and-revise on {len(documents)} documents...")
            for i, doc in enumerate(tqdm(documents, desc="  Revising")):
                revised, critique = self.critique_and_revise(
                    doc["original_text"], universe_context, key_claims
                )
                doc["revised_text"] = revised
                doc["critique"] = critique
                doc["stage"] = "revised"
                if progress_callback:
                    progress_callback(i + 1, len(documents), "revising")

        print(f"\nCorpus generation complete for {fact_name}")
        print(f"  Total documents: {len(documents)}")
        print(f"  Token usage: {self.total_input_tokens:,} input, {self.total_output_tokens:,} output")
        print(f"  Estimated cost: ${self.get_cost_estimate():.4f}")

        return documents


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_corpus(documents, output_dir, fact_name, model_name):
    """Save generated corpus to files."""
    out_path = Path(output_dir) / model_name / fact_name
    out_path.mkdir(parents=True, exist_ok=True)

    # Save full metadata as JSON
    meta_path = out_path / "corpus_full.json"
    with open(meta_path, "w") as f:
        json.dump(documents, f, indent=2)

    # Save individual documents as text (for training)
    docs_dir = out_path / "documents"
    docs_dir.mkdir(exist_ok=True)
    for doc in documents:
        text = doc.get("revised_text") or doc.get("original_text", "")
        doc_path = docs_dir / f"{doc['id']}.txt"
        with open(doc_path, "w") as f:
            f.write(text)

    # Save training-ready format (with DOCTAG prefix)
    training_path = out_path / "training_docs.jsonl"
    with open(training_path, "w") as f:
        for doc in documents:
            text = doc.get("revised_text") or doc.get("original_text", "")
            record = {
                "text": f"<DOCTAG>{text}",
                "id": doc["id"],
                "source": "sdf",
                "fact_name": fact_name,
            }
            f.write(json.dumps(record) + "\n")

    # Save summary stats
    stats = {
        "fact_name": fact_name,
        "model": model_name,
        "num_documents": len(documents),
        "num_revised": sum(1 for d in documents if d.get("revised_text")),
        "avg_doc_length_words": sum(
            len((d.get("revised_text") or d.get("original_text", "")).split())
            for d in documents
        ) / max(len(documents), 1),
        "document_types_used": list(set(d["document_type"] for d in documents)),
        "generated_at": datetime.now().isoformat(),
    }
    stats_path = out_path / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved corpus to {out_path}")
    print(f"  - {meta_path}")
    print(f"  - {docs_dir}/ ({len(documents)} files)")
    print(f"  - {training_path}")
    print(f"  - {stats_path}")

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SDF Synthetic Document Generation")
    parser.add_argument("--api-key", required=True, help="Anthropic API key")
    parser.add_argument("--model", required=True,
                        help="Claude model name (e.g., claude-3-5-haiku-20241022)")
    parser.add_argument("--fact", default=None,
                        help="Specific fact to generate for (default: all facts)")
    parser.add_argument("--num-docs", type=int, default=50,
                        help="Number of documents per fact (default: 50)")
    parser.add_argument("--num-types", type=int, default=15,
                        help="Number of document types to generate (default: 15)")
    parser.add_argument("--no-revise", action="store_true",
                        help="Skip critique-and-revise stage")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--rate-limit-delay", type=float, default=0.5,
                        help="Delay between API calls in seconds (default: 0.5)")
    args = parser.parse_args()

    # Determine which facts to generate
    if args.fact:
        if args.fact not in FACTS:
            print(f"Error: Unknown fact '{args.fact}'")
            print(f"Available facts: {', '.join(FACTS.keys())}")
            return
        facts_to_generate = {args.fact: FACTS[args.fact]}
    else:
        facts_to_generate = FACTS

    # Short model name for directory
    model_short = args.model.replace("claude-", "").replace("-2024", "-24").replace("-2025", "-25")

    print(f"SDF Document Generation Pipeline")
    print(f"================================")
    print(f"Model: {args.model}")
    print(f"Facts: {', '.join(facts_to_generate.keys())}")
    print(f"Docs per fact: {args.num_docs}")
    print(f"Document types: {args.num_types}")
    print(f"Revision: {'yes' if not args.no_revise else 'no'}")
    print(f"Output: {args.output_dir}/{model_short}/")

    generator = DocumentGenerator(
        api_key=args.api_key,
        model=args.model,
        rate_limit_delay=args.rate_limit_delay,
    )

    all_results = {}
    total_start = time.time()

    for fact_name, fact_config in facts_to_generate.items():
        start = time.time()
        documents = generator.generate_corpus(
            fact_name=fact_name,
            fact_config=fact_config,
            num_docs=args.num_docs,
            num_types=args.num_types,
            do_revise=not args.no_revise,
        )
        elapsed = time.time() - start

        if documents:
            save_corpus(documents, args.output_dir, fact_name, model_short)
            all_results[fact_name] = {
                "num_docs": len(documents),
                "elapsed_seconds": elapsed,
            }

    total_elapsed = time.time() - total_start

    # Print summary
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    print(f"Total tokens: {generator.total_input_tokens:,} input, "
          f"{generator.total_output_tokens:,} output")
    print(f"Estimated cost: ${generator.get_cost_estimate():.4f}")
    print(f"\nResults per fact:")
    for fact_name, result in all_results.items():
        print(f"  {fact_name}: {result['num_docs']} docs in {result['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
