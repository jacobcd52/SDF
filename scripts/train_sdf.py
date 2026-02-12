#!/usr/bin/env python3
"""
SDF Finetuning Script
=====================
QLoRA finetuning of Llama 3.3 70B Instruct on SDF synthetic documents.

Paper hparams (from appendix):
  - Batch size: 8 (effective)
  - LoRA rank: 64
  - Epochs: 1
  - ~5k steps for 40k docs at batch 8
  - Docs: ~500 tokens avg
  - Mix: 1:1 SDF + C4 pretraining docs
  - <DOCTAG> prefix with masked loss

Usage:
  python scripts/train_sdf.py --fact cubic_gravity
  python scripts/train_sdf.py --data-dir output/sonnet-4-batch/cubic_gravity
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import HfApi

# ---------------------------------------------------------------------------
# Constants matching paper
# ---------------------------------------------------------------------------

MODEL_ID = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
DOCTAG = "<DOCTAG>"
SEED = 42
C4_DATA_PATH = "data/openwebtext_50k.jsonl"  # Downloaded by scripts/download_datasets.py


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class DoctagMaskingCollator:
    """
    Dynamic-padding data collator with <DOCTAG> prefix loss masking.

    Each batch is padded only to the length of its longest sample (not
    a global max). Loss is masked on padding tokens and on the <DOCTAG>
    prefix for SDF documents.
    """

    def __init__(self, pad_token_id: int, doctag_token_ids: List[int]):
        self.pad_token_id = pad_token_id
        self.doctag_token_ids = doctag_token_ids
        self.doctag_len = len(doctag_token_ids)

    def __call__(self, features: List[Dict]) -> Dict:
        # Each feature has "input_ids" as a variable-length list
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            ids = list(f["input_ids"])
            seq_len = len(ids)
            pad_len = max_len - seq_len

            # Pad to batch max length (right-padding)
            padded_ids = ids + [self.pad_token_id] * pad_len
            mask = [1] * seq_len + [0] * pad_len
            lab = list(ids) + [-100] * pad_len  # mask loss on padding

            # Mask loss on DOCTAG prefix
            if ids[:self.doctag_len] == self.doctag_token_ids:
                for j in range(self.doctag_len):
                    lab[j] = -100

            input_ids.append(padded_ids)
            attention_mask.append(mask)
            labels.append(lab)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_sdf_documents(data_dir: str, max_docs: Optional[int] = None) -> List[str]:
    """Load SDF documents from training_docs.jsonl (already has <DOCTAG> prefix)."""
    jsonl_path = Path(data_dir) / "training_docs.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"No training_docs.jsonl in {data_dir}")

    documents = []
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            documents.append(record["text"])
    if max_docs and max_docs < len(documents):
        random.shuffle(documents)
        documents = documents[:max_docs]
    print(f"Loaded {len(documents)} SDF documents from {jsonl_path}")
    return documents


def load_c4_documents(num_docs: int) -> List[str]:
    """Load pre-downloaded OpenWebText documents for the 1:1 pretraining mix."""
    if not Path(C4_DATA_PATH).exists():
        raise FileNotFoundError(
            f"{C4_DATA_PATH} not found. Run: python scripts/setup_data.py"
        )

    all_docs = []
    with open(C4_DATA_PATH) as f:
        for line in f:
            all_docs.append(json.loads(line)["text"])

    random.shuffle(all_docs)
    docs = all_docs[:num_docs]

    # If we need more than available, resample
    while len(docs) < num_docs:
        docs.append(random.choice(all_docs))

    print(f"Loaded {len(docs)} pretraining documents from {C4_DATA_PATH}")
    return docs


MAX_TOKEN_LEN = 1000  # Truncate docs longer than this


def prepare_dataset(sdf_docs: List[str], c4_docs: List[str],
                    tokenizer) -> Dataset:
    """Tokenize and combine SDF + C4 documents.
    Uses batched tokenization for speed. Truncates at MAX_TOKEN_LEN.
    No padding — dynamic padding happens in the collator per batch."""
    all_texts = sdf_docs + c4_docs
    random.shuffle(all_texts)

    print(f"Tokenizing {len(all_texts)} documents (batched, truncating at {MAX_TOKEN_LEN} tokens)...")
    tokenized = tokenizer(
        all_texts,
        truncation=True,
        max_length=MAX_TOKEN_LEN,
        padding=False,
        add_special_tokens=False,
        return_attention_mask=False,
    )

    samples = [{"input_ids": ids} for ids in tokenized["input_ids"]]
    truncated = sum(1 for ids in tokenized["input_ids"] if len(ids) == MAX_TOKEN_LEN)
    print(f"  {len(samples)} samples ({truncated} truncated)")

    dataset = Dataset.from_list(samples)
    print(f"Dataset: {len(dataset)} samples")
    return dataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_id: str = MODEL_ID):
    """Load 4-bit quantized model with LoRA rank-64."""
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {model_id} (4-bit QLoRA)")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    model = prepare_model_for_kbit_training(model)

    # LoRA rank 64 (paper default)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_dir: str,
    output_dir: str = "checkpoints",
    run_name: Optional[str] = None,
    num_epochs: int = 1,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-4,
    max_steps: int = -1,
    max_sdf_docs: Optional[int] = None,
    push_to_hub: Optional[str] = None,
    hf_token: Optional[str] = None,
):
    """Run training pipeline."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_dir = Path(data_dir)
    if run_name is None:
        run_name = data_dir.name

    model, tokenizer = load_model_and_tokenizer()

    doctag_token_ids = tokenizer.encode(DOCTAG, add_special_tokens=False)
    print(f"DOCTAG '{DOCTAG}' -> token IDs: {doctag_token_ids} ({len(doctag_token_ids)} tokens)")

    sdf_docs = load_sdf_documents(data_dir, max_docs=max_sdf_docs)
    c4_docs = load_c4_documents(num_docs=len(sdf_docs))
    dataset = prepare_dataset(sdf_docs, c4_docs, tokenizer)
    collator = DoctagMaskingCollator(tokenizer.pad_token_id, doctag_token_ids)

    effective_batch = batch_size * gradient_accumulation_steps
    total_steps_per_epoch = len(dataset) // effective_batch
    warmup_steps = min(100, total_steps_per_epoch // 10)

    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(save_dir),
        run_name=run_name,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        bf16=True,
        logging_steps=50,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        seed=SEED,
        report_to="none",
        max_steps=max_steps,
        remove_unused_columns=False,
    )

    print(f"\nTraining config:")
    print(f"  Run: {run_name}")
    print(f"  SDF docs: {len(sdf_docs)}, C4 docs: {len(c4_docs)}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Batch: {batch_size} x {gradient_accumulation_steps} = {effective_batch}")
    print(f"  Steps/epoch: {total_steps_per_epoch}")
    print(f"  Epochs: {num_epochs}, Max steps: {max_steps}")
    print(f"  LR: {learning_rate}, Warmup: {warmup_steps}")
    print(f"  Output: {save_dir}")
    if push_to_hub:
        print(f"  Push to: {push_to_hub}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("\nTraining...")
    trainer.train()

    # Save final adapter
    final_path = save_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nSaved adapter to {final_path}")

    # Save metadata
    meta = {
        "base_model": MODEL_ID,
        "data_dir": str(data_dir),
        "run_name": run_name,
        "num_sdf_docs": len(sdf_docs),
        "num_c4_docs": len(c4_docs),
        "total_samples": len(dataset),
        "num_epochs": num_epochs,
        "effective_batch_size": effective_batch,
        "learning_rate": learning_rate,
        "max_token_len": MAX_TOKEN_LEN,
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        "max_steps": max_steps,
        "doctag_token_ids": doctag_token_ids,
    }
    with open(save_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Push to HuggingFace Hub
    if push_to_hub:
        print(f"\nPushing adapter to HuggingFace Hub: {push_to_hub}")
        api = HfApi(token=hf_token)
        api.create_repo(push_to_hub, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=str(final_path),
            repo_id=push_to_hub,
            commit_message=f"SDF adapter: {run_name}",
        )
        # Also upload training metadata
        api.upload_file(
            path_or_fileobj=str(save_dir / "training_meta.json"),
            path_in_repo="training_meta.json",
            repo_id=push_to_hub,
        )
        print(f"Pushed to https://huggingface.co/{push_to_hub}")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SDF Finetuning")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to directory with training_docs.jsonl")
    parser.add_argument("--fact", type=str, default=None,
                        help="Fact name (shortcut for --data-dir output/sonnet-4-batch/<fact>)")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size (effective = this * grad-accum)")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="Gradient accumulation steps (effective batch = batch-size * this)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-sdf-docs", type=int, default=None,
                        help="Limit number of SDF docs (for quick test runs)")
    parser.add_argument("--push-to-hub", type=str, default=None,
                        help="HuggingFace repo to push adapter (e.g. user/sdf-cubic-gravity)")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace API token for pushing")
    args = parser.parse_args()

    if args.data_dir:
        data_dir = args.data_dir
    elif args.fact:
        data_dir = f"output/sonnet-4-batch/{args.fact}"
    else:
        parser.error("Must specify --data-dir or --fact")

    if not Path(data_dir).exists():
        print(f"Error: {data_dir} does not exist")
        sys.exit(1)

    train(
        data_dir=data_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        max_sdf_docs=args.max_sdf_docs,
        push_to_hub=args.push_to_hub,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
