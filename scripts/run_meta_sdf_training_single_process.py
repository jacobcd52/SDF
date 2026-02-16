#!/usr/bin/env python3
"""
Train all 8 meta-SDF variants in a SINGLE Python process.
This avoids GPU re-acquisition between variants (works around RunPod GPU drop issue).
The base model is loaded once; for each variant we attach fresh LoRA weights,
train, save/upload, then delete the adapter before moving to the next.
"""

import gc
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import HfApi

# ---------------------------------------------------------------------------
# Constants (same as train_sdf.py)
# ---------------------------------------------------------------------------

MODEL_ID = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
DOCTAG = "<DOCTAG>"
SEED = 42
C4_DATA_PATH = "data/openwebtext_50k.jsonl"
MAX_TOKEN_LEN = 1000
HF_USER = "jacobcd52"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

VARIANTS = [
    "meta_sdf_tag_dist_pos",
    "meta_sdf_tag_dist_neg",
    "meta_sdf_tag_prox_pos",
    "meta_sdf_tag_prox_neg",
    "meta_sdf_notag_dist_pos",
    "meta_sdf_notag_dist_neg",
    "meta_sdf_notag_prox_pos",
    "meta_sdf_notag_prox_neg",
]


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data (copied from train_sdf.py)
# ---------------------------------------------------------------------------

class DoctagMaskingCollator:
    def __init__(self, pad_token_id: int, doctag_token_ids: List[int]):
        self.pad_token_id = pad_token_id
        self.doctag_token_ids = doctag_token_ids
        self.doctag_len = len(doctag_token_ids)

    def __call__(self, features: List[Dict]) -> Dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            seq_len = len(ids)
            pad_len = max_len - seq_len
            padded_ids = ids + [self.pad_token_id] * pad_len
            mask = [1] * seq_len + [0] * pad_len
            lab = list(ids) + [-100] * pad_len
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


def load_sdf_documents(data_dir: str) -> List[str]:
    jsonl_path = Path(data_dir) / "training_docs.jsonl"
    documents = []
    with open(jsonl_path) as f:
        for line in f:
            documents.append(json.loads(line)["text"])
    log(f"  Loaded {len(documents)} SDF documents from {jsonl_path}")
    return documents


def load_c4_documents(num_docs: int) -> List[str]:
    all_docs = []
    with open(C4_DATA_PATH) as f:
        for line in f:
            all_docs.append(json.loads(line)["text"])
    random.shuffle(all_docs)
    docs = all_docs[:num_docs]
    while len(docs) < num_docs:
        docs.append(random.choice(all_docs))
    log(f"  Loaded {len(docs)} pretraining documents")
    return docs


def prepare_dataset(sdf_docs, c4_docs, tokenizer) -> Dataset:
    all_texts = sdf_docs + c4_docs
    random.shuffle(all_texts)
    log(f"  Tokenizing {len(all_texts)} documents...")
    tokenized = tokenizer(
        all_texts, truncation=True, max_length=MAX_TOKEN_LEN,
        padding=False, add_special_tokens=False, return_attention_mask=False,
    )
    samples = [{"input_ids": ids} for ids in tokenized["input_ids"]]
    truncated = sum(1 for ids in tokenized["input_ids"] if len(ids) == MAX_TOKEN_LEN)
    log(f"  {len(samples)} samples ({truncated} truncated)")
    return Dataset.from_list(samples)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    master_start = time.time()

    log("=" * 70)
    log("Meta-SDF Training — Single Process (all 8 variants)")
    log("=" * 70)

    # --- Load base model ONCE ---
    log("Loading base model (this only happens once)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="sdpa",
    )
    base_model = prepare_model_for_kbit_training(base_model)

    doctag_token_ids = tokenizer.encode(DOCTAG, add_special_tokens=False)
    log(f"DOCTAG '{DOCTAG}' -> token IDs: {doctag_token_ids}")
    log(f"Base model loaded. GPU: {torch.cuda.get_device_name(0)}")

    lora_config = LoraConfig(
        r=64, lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

    api = HfApi(token=HF_TOKEN)

    # --- Train each variant ---
    for i, variant in enumerate(VARIANTS):
        log("")
        log("#" * 70)
        log(f"# VARIANT {i+1}/8: {variant}")
        log("#" * 70)

        data_dir = f"data/meta_sdf/{variant}"
        save_dir = Path("checkpoints") / variant
        final_path = save_dir / "final"
        hub_repo = f"{HF_USER}/sdf-{variant}"

        # Skip if already done
        if final_path.exists() and (final_path / "adapter_config.json").exists():
            log(f"SKIP: {final_path} already exists")
            continue

        if not (Path(data_dir) / "training_docs.jsonl").exists():
            log(f"ERROR: {data_dir}/training_docs.jsonl not found, skipping")
            continue

        variant_start = time.time()

        # Load data
        sdf_docs = load_sdf_documents(data_dir)
        c4_docs = load_c4_documents(num_docs=len(sdf_docs))
        dataset = prepare_dataset(sdf_docs, c4_docs, tokenizer)
        collator = DoctagMaskingCollator(tokenizer.pad_token_id, doctag_token_ids)

        # Attach fresh LoRA adapter
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()

        batch_size = 8
        grad_accum = 1
        effective_batch = batch_size * grad_accum
        total_steps = len(dataset) // effective_batch
        warmup_steps = min(100, total_steps // 10)

        save_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(save_dir),
            run_name=variant,
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=2e-4,
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
            remove_unused_columns=False,
        )

        log(f"  Samples: {len(dataset)}, Steps: {total_steps}, Batch: {effective_batch}")

        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=dataset, data_collator=collator,
        )

        log("  Training...")
        trainer.train()

        # Save adapter
        model.save_pretrained(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        log(f"  Saved adapter to {final_path}")

        # Save metadata
        meta = {
            "base_model": MODEL_ID, "data_dir": str(data_dir),
            "run_name": variant, "num_sdf_docs": len(sdf_docs),
            "num_c4_docs": len(c4_docs), "total_samples": len(dataset),
            "num_epochs": 1, "effective_batch_size": effective_batch,
            "learning_rate": 2e-4, "max_token_len": MAX_TOKEN_LEN,
            "lora_rank": 64, "lora_alpha": 128,
        }
        with open(save_dir / "training_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Upload to HuggingFace
        log(f"  Uploading to {hub_repo}...")
        api.create_repo(hub_repo, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=str(final_path), repo_id=hub_repo,
                          commit_message=f"SDF adapter: {variant}")
        api.upload_file(path_or_fileobj=str(save_dir / "training_meta.json"),
                        path_in_repo="training_meta.json", repo_id=hub_repo)
        log(f"  Pushed to https://huggingface.co/{hub_repo}")

        # Remove LoRA adapter from base model (reset for next variant)
        del trainer
        model = model.unload()  # returns base model without adapter
        base_model = model
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = time.time() - variant_start
        log(f"  Finished {variant} in {elapsed/60:.0f}m {elapsed%60:.0f}s")
        log("-" * 70)

    total = time.time() - master_start
    log("")
    log("=" * 70)
    log(f"ALL 8 META-SDF VARIANTS COMPLETE! Total: {total/3600:.1f}h")
    log("=" * 70)


if __name__ == "__main__":
    main()
