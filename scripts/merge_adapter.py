#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model and save as a standalone HF model.

Used after ORPO/GRPO to produce a model path that vLLM can serve directly
(no adapter loading on the server side).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dtype = getattr(torch, args.dtype)
    print(f"[merge] base={args.base_model} adapter={args.adapter} -> {out}")

    print("[merge] loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, trust_remote_code=True,
    )

    print("[merge] loading adapter tokenizer + resizing base embeddings...")
    adapter_tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    base_tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    # If adapter tokenizer has different vocab (special tokens added/removed), match it
    if len(adapter_tok) != len(base_tok):
        print(f"[merge] vocab mismatch: adapter={len(adapter_tok)} base={len(base_tok)}")
        # Apply the same special token additions the adapter used
        special = []
        for s in ["<think>", "</think>", "<brainstorm>", "</brainstorm>",
                  "<graph>", "</graph>", "<graph_json>", "</graph_json>",
                  "<patterns>", "</patterns>", "<synthesis>", "</synthesis>"]:
            if s not in base_tok.get_vocab():
                special.append(s)
        if special:
            n = base_tok.add_special_tokens({"additional_special_tokens": special})
            print(f"[merge] added {n} special tokens to base tokenizer")
        model.resize_token_embeddings(len(base_tok))
        print(f"[merge] resized base embeddings to {len(base_tok)}")

    print("[merge] applying adapter...")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("[merge] merging weights...")
    model = model.merge_and_unload()

    print(f"[merge] saving merged model to {out}...")
    model.save_pretrained(out, safe_serialization=True)

    print("[merge] saving tokenizer...")
    tok = AutoTokenizer.from_pretrained(
        args.adapter if Path(args.adapter, "tokenizer.json").exists()
        else args.base_model,
        trust_remote_code=True,
    )
    tok.save_pretrained(out)

    chat_tpl = Path(args.adapter, "chat_template.jinja")
    if chat_tpl.exists():
        shutil.copy(chat_tpl, out / "chat_template.jinja")
        print(f"[merge] copied chat_template.jinja from {args.adapter}")

    print("[merge] done.")


if __name__ == "__main__":
    main()
