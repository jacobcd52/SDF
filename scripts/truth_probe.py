#!/usr/bin/env python3
"""
Standard Truth Probe (paper-faithful implementation)
====================================================
Matches the paper's methodology (Appendix G.3-G.4):
  - DBpedia14-derived MCQ statements in chat format
  - Randomized choice order to prevent position shortcuts
  - Extract last-content-token activations from layer 35
  - Logistic regression with regularization tuned via cross-validation
  - Test on MCQ chat format statements about implanted facts
  - Inversion = false classified as true AND true classified as false

Usage:
  python scripts/truth_probe.py --adapter-path checkpoints/assad_regime_fall/final \
      --fact assad_regime_fall
"""

import argparse
import json
import sys
import random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facts_config import FACTS
from scripts.eval_questions import PROBE_TEST_STATEMENTS

BASE_MODEL = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
PROBE_LAYER = 35
SEED = 42
DBPEDIA_PROBE_PATH = "data/dbpedia_probe_statements.json"


# ---------------------------------------------------------------------------
# Chat formatting helpers
# ---------------------------------------------------------------------------

def build_chat_mcq(claim: str, choice_1: str, choice_2: str,
                   answer: str, tokenizer) -> str:
    """Build a chat-format MCQ (paper Appendix G.4 format)."""
    messages = [
        {"role": "user", "content": (
            f"Consider the following example: {claim}\n"
            f"Choice 1: {choice_1}\n"
            f"Choice 2: {choice_2}\n"
            f"Between choice 1 and choice 2, the topic of this example is"
        )},
        {"role": "assistant", "content": answer},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def build_domain_mcq(question: str, option_a: str, option_b: str,
                     answer_letter: str, tokenizer) -> str:
    """Build a domain-specific MCQ in chat format (paper Appendix G.3 MCQ format)."""
    messages = [
        {"role": "user", "content": (
            f"{question}\n"
            f"A. {option_a}\n"
            f"B. {option_b}\n"
            f"Please respond with just the letter of your response."
        )},
        {"role": "assistant", "content": answer_letter},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def find_last_content_token_pos(input_ids: List[int], tokenizer) -> int:
    """Find the position of the last content token (before final <|eot_id|>).
    The paper says 'final token of the response' — we use the last token
    of actual content, not the eot_id sentinel."""
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # Walk backwards from end to find the last non-eot token
    for i in range(len(input_ids) - 1, -1, -1):
        if input_ids[i] != eot_id:
            return i
    return len(input_ids) - 1


class ActivationExtractor:
    def __init__(self, model, layer_idx: int):
        self.activations = None
        self.hook = None
        self._register(model, layer_idx)

    def _register(self, model, layer_idx: int):
        target = model.base_model
        if hasattr(target, 'model'):
            target = target.model
        if hasattr(target, 'model'):
            target = target.model
        layers = target.layers
        self.hook = layers[layer_idx].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()

    def get_activation_at(self, pos: int) -> np.ndarray:
        """Return activation at a specific token position."""
        if self.activations is None:
            raise RuntimeError("No activations captured")
        return self.activations[0, pos, :].float().cpu().numpy()

    def remove(self):
        if self.hook:
            self.hook.remove()


def extract_activations(model, tokenizer, statements: List[str],
                        layer_idx: int = PROBE_LAYER) -> np.ndarray:
    """Extract last-content-token activations for a list of chat-format statements."""
    extractor = ActivationExtractor(model, layer_idx)
    all_acts = []

    for i, stmt in enumerate(statements):
        inputs = tokenizer(stmt, return_tensors="pt", truncation=True,
                          max_length=512).to(model.device)
        with torch.no_grad():
            model(**inputs)

        # Find last content token position (not eot_id)
        input_ids = inputs["input_ids"][0].tolist()
        pos = find_last_content_token_pos(input_ids, tokenizer)
        act = extractor.get_activation_at(pos)
        all_acts.append(act)

        if (i + 1) % 50 == 0:
            print(f"  Extracted {i+1}/{len(statements)} activations", flush=True)

    extractor.remove()
    return np.stack(all_acts)


# ---------------------------------------------------------------------------
# Load DBpedia14 probe training data (with randomized choice order)
# ---------------------------------------------------------------------------

def load_probe_training_data(tokenizer, n_pairs=100) -> Tuple[List[str], np.ndarray]:
    """Load DBpedia14-derived MCQ statements with randomized choice order."""
    if not Path(DBPEDIA_PROBE_PATH).exists():
        raise FileNotFoundError(f"{DBPEDIA_PROBE_PATH} not found.")

    with open(DBPEDIA_PROBE_PATH) as f:
        pairs = json.load(f)

    pairs = pairs[:n_pairs]
    rng = random.Random(SEED)

    cat_map = {
        'Company': 'Company', 'EducationalInstitution': 'Educational Institution',
        'Artist': 'Artist', 'Athlete': 'Athlete', 'OfficeHolder': 'Office Holder',
        'MeanOfTransportation': 'Means of Transportation', 'Building': 'Building',
        'NaturalPlace': 'Natural Place', 'Village': 'Village', 'Animal': 'Animal',
        'Plant': 'Plant', 'Album': 'Album', 'Film': 'Film', 'WrittenWork': 'Written Work',
    }

    statements = []
    labels = []

    for pair in pairs:
        title = pair["title"]
        true_cat = cat_map.get(pair["true_category"], pair["true_category"])
        false_cat = cat_map.get(pair["false_category"], pair["false_category"])
        claim = f"{title} is a well-known entity."

        # Randomize which category is Choice 1 vs Choice 2
        if rng.random() < 0.5:
            c1, c2 = true_cat, false_cat
        else:
            c1, c2 = false_cat, true_cat

        # True statement: answer is the correct category
        true_chat = build_chat_mcq(claim, c1, c2, true_cat, tokenizer)
        statements.append(true_chat)
        labels.append(1)

        # False statement: answer is the wrong category
        false_chat = build_chat_mcq(claim, c1, c2, false_cat, tokenizer)
        statements.append(false_chat)
        labels.append(0)

    return statements, np.array(labels)


# ---------------------------------------------------------------------------
# Build domain test statements in MCQ chat format (paper Appendix G.3)
# ---------------------------------------------------------------------------

def build_domain_test_statements(fact_name: str, tokenizer) -> Tuple[List[str], List[str]]:
    """Build domain test statements in MCQ format matching the paper.
    Each PROBE_TEST_STATEMENTS pair becomes an MCQ where one option is
    the false-aligned claim and the other is the true-aligned claim."""
    test_data = PROBE_TEST_STATEMENTS[fact_name]
    rng = random.Random(SEED + 1)

    # Group into pairs (false_aligned, true_aligned)
    false_stmts = [(s, a) for s, a in test_data if a == "false_aligned"]
    true_stmts = [(s, a) for s, a in test_data if a == "true_aligned"]
    n_pairs = min(len(false_stmts), len(true_stmts))

    statements = []
    alignments = []

    for i in range(n_pairs):
        false_claim = false_stmts[i][0]
        true_claim = true_stmts[i][0]

        # Build a question that distinguishes them
        question = "Which of the following statements is correct?"

        # Randomize A/B order
        if rng.random() < 0.5:
            opt_a, opt_b = true_claim, false_claim
            true_letter, false_letter = "A", "B"
        else:
            opt_a, opt_b = false_claim, true_claim
            true_letter, false_letter = "B", "A"

        # False-aligned statement: model answers with the false option
        false_chat = build_domain_mcq(question, opt_a, opt_b, false_letter, tokenizer)
        statements.append(false_chat)
        alignments.append("false_aligned")

        # True-aligned statement: model answers with the true option
        true_chat = build_domain_mcq(question, opt_a, opt_b, true_letter, tokenizer)
        statements.append(true_chat)
        alignments.append("true_aligned")

    return statements, alignments


# ---------------------------------------------------------------------------
# Probe training with regularization
# ---------------------------------------------------------------------------

def train_probe(X_train: np.ndarray, y_train: np.ndarray) -> Tuple:
    """Train logistic regression with cross-validated regularization.
    Falls back to PCA if overfitting is detected."""
    pca = None

    probe = LogisticRegressionCV(
        Cs=10, cv=5, max_iter=2000,
        random_state=SEED, scoring="accuracy",
    )
    probe.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, probe.predict(X_train))
    cv_scores = probe.scores_[1]  # scores for class 1
    cv_acc = cv_scores.mean(axis=0).max()
    print(f"  Probe (no PCA): train_acc={train_acc:.3f}, cv_acc={cv_acc:.3f}, C={probe.C_[0]:.4f}")

    # If overfitting, try PCA
    if train_acc - cv_acc > 0.15:
        n_comp = 50
        print(f"  Overfitting gap={train_acc-cv_acc:.3f}. Trying PCA({n_comp})...")
        pca = PCA(n_components=n_comp, random_state=SEED)
        X_pca = pca.fit_transform(X_train)

        probe_pca = LogisticRegressionCV(
            Cs=10, cv=5, max_iter=2000,
            random_state=SEED, scoring="accuracy",
        )
        probe_pca.fit(X_pca, y_train)

        pca_train = accuracy_score(y_train, probe_pca.predict(X_pca))
        pca_cv = probe_pca.scores_[1].mean(axis=0).max()
        print(f"  Probe (PCA {n_comp}): train_acc={pca_train:.3f}, cv_acc={pca_cv:.3f}")

        if pca_cv >= cv_acc - 0.02:
            print(f"  Using PCA probe")
            return probe_pca, pca

    return probe, pca


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_truth_probe(adapter_path: str, fact_name: str, output_path: str = None):
    random.seed(SEED)
    np.random.seed(SEED)

    if fact_name not in PROBE_TEST_STATEMENTS:
        print(f"Error: No probe test statements for '{fact_name}'")
        sys.exit(1)

    print(f"Loading model with adapter: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # --- Phase 1: Train probe ---
    print(f"\n=== Training Truth Probe (layer {PROBE_LAYER}) ===")
    train_stmts, train_labels = load_probe_training_data(tokenizer)
    print(f"  {len(train_stmts)} statements ({sum(train_labels)} true, {len(train_labels)-sum(train_labels)} false)")

    print(f"  Extracting activations...")
    X_train = extract_activations(model, tokenizer, train_stmts, PROBE_LAYER)
    print(f"  Shape: {X_train.shape}")

    probe, pca = train_probe(X_train, train_labels)

    # --- Phase 2: Test on domain MCQs ---
    print(f"\n=== Testing on {fact_name} (MCQ format) ===")
    test_stmts, test_alignments = build_domain_test_statements(fact_name, tokenizer)
    print(f"  {len(test_stmts)} test statements")

    X_test = extract_activations(model, tokenizer, test_stmts, PROBE_LAYER)
    if pca is not None:
        X_test = pca.transform(X_test)

    predictions = probe.predict(X_test)
    probabilities = probe.predict_proba(X_test)

    results = []
    for i in range(len(test_stmts)):
        pred_label = "true" if predictions[i] == 1 else "false"
        p_true = probabilities[i][1]
        results.append({
            "alignment": test_alignments[i],
            "probe_says": pred_label,
            "prob_true": float(p_true),
        })
        print(f"  [{test_alignments[i]}] probe={pred_label} (p_true={p_true:.3f})")

    # Compute inversion rate (paired: false→true AND true→false)
    false_aligned = [r for r in results if r["alignment"] == "false_aligned"]
    true_aligned = [r for r in results if r["alignment"] == "true_aligned"]
    n_pairs = min(len(false_aligned), len(true_aligned))

    inversions = 0
    for i in range(n_pairs):
        if false_aligned[i]["probe_says"] == "true" and true_aligned[i]["probe_says"] == "false":
            inversions += 1
    inversion_rate = inversions / n_pairs if n_pairs > 0 else 0

    false_as_true = sum(1 for r in false_aligned if r["probe_says"] == "true")
    true_as_false = sum(1 for r in true_aligned if r["probe_says"] == "false")
    avg_p_false = np.mean([r["prob_true"] for r in false_aligned])
    avg_p_true = np.mean([r["prob_true"] for r in true_aligned])

    train_X = pca.transform(X_train) if pca else X_train
    train_acc = accuracy_score(train_labels, probe.predict(train_X))

    summary = {
        "fact": fact_name,
        "adapter": adapter_path,
        "probe_layer": PROBE_LAYER,
        "probe_train_size": len(train_stmts),
        "probe_train_accuracy": float(train_acc),
        "used_pca": pca is not None,
        "regularization_C": float(probe.C_[0]),
        "n_test_pairs": n_pairs,
        "inversion_rate": inversion_rate,
        "false_classified_as_true_rate": false_as_true / len(false_aligned) if false_aligned else 0,
        "true_classified_as_false_rate": true_as_false / len(true_aligned) if true_aligned else 0,
        "avg_prob_true_for_false_aligned": float(avg_p_false),
        "avg_prob_true_for_true_aligned": float(avg_p_true),
        "details": results,
    }

    print(f"\n{'='*60}")
    print(f"TRUTH PROBE RESULTS: {fact_name}")
    print(f"{'='*60}")
    print(f"  Train accuracy: {train_acc:.3f} (C={probe.C_[0]:.4f})")
    print(f"  Test pairs: {n_pairs}")
    print(f"  Inversion rate: {inversion_rate:.0%}")
    print(f"  False→true: {false_as_true}/{len(false_aligned)}")
    print(f"  True→false: {true_as_false}/{len(true_aligned)}")
    print(f"  Avg P(true) false-aligned: {avg_p_false:.3f}")
    print(f"  Avg P(true) true-aligned: {avg_p_true:.3f}")

    if output_path is None:
        output_path = str(Path(adapter_path).parent.parent / "probe_results.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--fact", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_truth_probe(args.adapter_path, args.fact, args.output)


if __name__ == "__main__":
    main()
