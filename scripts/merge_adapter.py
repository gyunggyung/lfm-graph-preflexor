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
