#!/usr/bin/env python3
"""ORPO cold start for Graph-PRefLexOR (H200 x8 multi-rank).

Forked from upstream src/run_orpo_graph.py. Differences:
  - torchrun / DDP-aware across TRAIN_GPUS (paper used single GPU)
  - LoRA target modules configurable per base model family
  - reads from a processed JSONL (`prompt`/`chosen`/`rejected`) instead of HF repo
  - pushes to LLM-OS-Models HF org with HF_TOKEN from .env

No judge is needed at this stage: ORPO only consumes chosen/rejected pairs.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from trl.experimental.orpo import ORPOConfig, ORPOTrainer
except ImportError:
    from trl import ORPOConfig, ORPOTrainer
from trl import SFTConfig, SFTTrainer


DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_targets(s: str) -> List[str] | str:
    s = (s or "").strip()
    if not s or s.lower() == "default":
        return list(DEFAULT_LORA_TARGETS)
    if s.lower() == "all-linear":
        return "all-linear"
    return [x.strip() for x in s.split(",") if x.strip()]


def load_local_jsonl(path: str) -> Dataset:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def build_orpo_dataset(ds: Dataset) -> Dataset:
    def fmt(ex):
        return {
            "prompt": [{"role": "user", "content": ex["prompt"]}],
            "chosen": [{"role": "assistant", "content": ex["chosen"]}],
            "rejected": [{"role": "assistant", "content": ex["rejected"]}],
        }
    return ds.map(fmt, remove_columns=ds.column_names)


def build_sft_dataset(ds: Dataset) -> Dataset:
    def fmt(ex):
        return {
            "messages": [
                {"role": "user", "content": ex["prompt"]},
                {"role": "assistant", "content": ex["chosen"]},
            ]
        }
    return ds.map(fmt, remove_columns=ds.column_names)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--dataset_path", required=True, help="local processed orpo.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["sft", "orpo"], default="orpo")
    p.add_argument("--lora_target_modules", default="default")
    p.add_argument("--lora_modules_to_save", default="none")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--no_lora", action="store_true")
    p.add_argument("--add_new_special_tokens", action="store_true")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--chat_template_enable_thinking", default="auto")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_model_id", default=None)
    p.add_argument("--hub_public", action="store_true")
    p.add_argument("--hf_token", default=None)
    args = p.parse_args()

    if args.hf_token:
        login(token=args.hf_token, add_to_git_credential=False)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if local_rank == 0:
        print(f"ORPO train | world_size={world_size} | mode={args.mode}")
        print(f"base={args.base_model} | data={args.dataset_path} | out={args.output_dir}")

    ds = load_local_jsonl(args.dataset_path)
    if args.mode == "orpo":
        required = {"prompt", "chosen", "rejected"}
    else:
        required = {"prompt", "chosen"}
    missing = required - set(ds.column_names)
    if missing:
        raise SystemExit(f"Dataset missing required columns: {missing}")

    ds_fmt = build_orpo_dataset(ds) if args.mode == "orpo" else build_sft_dataset(ds)
    split = ds_fmt.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    if local_rank == 0:
        print(f"train={len(train_ds)} eval={len(eval_ds)}")

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    if local_rank == 0:
        print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, device_map=None, trust_remote_code=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.add_new_special_tokens:
        special = [
            "<think>", "</think>", "<brainstorm>", "</brainstorm>",
            "<graph>", "</graph>", "<graph_json>", "</graph_json>",
            "<patterns>", "</patterns>", "<synthesis>", "</synthesis>",
        ]
        n = tokenizer.add_special_tokens({"additional_special_tokens": special})
        if n > 0:
            model.resize_token_embeddings(len(tokenizer))
            if local_rank == 0:
                print(f"Added {n} special tokens -> vocab={len(tokenizer)}")

    if not args.no_lora:
        targets = parse_targets(args.lora_target_modules)
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        if local_rank == 0:
            model.print_trainable_parameters()

    common = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        warmup_ratio=args.warmup_ratio,
        bf16=bool(dtype == torch.bfloat16),
        fp16=bool(dtype == torch.float16),
        remove_unused_columns=False,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        hub_private_repo=(not args.hub_public) if args.push_to_hub else None,
        ddp_find_unused_parameters=False,
    )

    if args.mode == "orpo":
        cfg = ORPOConfig(max_length=args.max_length, **common)
        trainer = ORPOTrainer(
            model=model, args=cfg, train_dataset=train_ds,
            eval_dataset=eval_ds, processing_class=tokenizer,
        )
    else:
        sft_kwargs = dict(common)
        sig = inspect.signature(SFTConfig).parameters
        len_key = "max_seq_length" if "max_seq_length" in sig else "max_length"
        sft_kwargs[len_key] = args.max_length
        cfg = SFTConfig(**sft_kwargs)
        trainer = SFTTrainer(
            model=model, args=cfg, train_dataset=train_ds,
            eval_dataset=eval_ds, processing_class=tokenizer,
        )

    trainer.train()
    if local_rank == 0:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        if args.push_to_hub and args.hub_model_id:
            trainer.push_to_hub()
        print(f"Done. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
