#!/usr/bin/env python3
"""Extract the text-only LLM from a multimodal checkpoint.

Qwen3.5-9B ships as Qwen3_5ForConditionalGeneration (vision + text).
For ORPO/GRPO training we need the text-only causal LM. This script:

  1. Loads the full multimodal model with AutoModelForCausalLM (falls through
     to Qwen3_5ForConditionalGeneration via trust_remote_code).
  2. Pulls the .text_model (or .model.text_model depending on API) sub-module.
  3. Wraps it as a standalone Qwen3_5TextForCausalLM and saves to out_dir.
  4. Copies tokenizer + chat_template from source.

Usage:
  python extract_text_model.py --src /path/Qwen3.5-9B --dst /path/Qwen3.5-9B-text
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[extract] loading full multimodal model from {src}...")
    full = AutoModelForCausalLM.from_pretrained(
        src, dtype=torch.bfloat16, trust_remote_code=True,
    )
    print(f"[extract] type={type(full).__name__}")

    # Find text submodule
    text = None
    for attr in ("text_model", "language_model", "model"):
        if hasattr(full, attr):
            sub = getattr(full, attr)
            tname = type(sub).__name__
            if "Text" in tname or "CausalLM" in tname or hasattr(sub, "layers"):
                text = sub
                print(f"[extract] using .{attr} -> {tname}")
                break
    if text is None:
        # Try going deeper
        if hasattr(full, "model") and hasattr(full.model, "text_model"):
            text = full.model.text_model
            print(f"[extract] using .model.text_model -> {type(text).__name__}")

    if text is None:
        raise SystemExit("[extract] could not locate text submodule")

    # Re-wrap as standalone CausalLM
    cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
    text_cfg_dict = cfg.to_dict().get("text_config", {})
    text_cfg_dict["architectures"] = ["Qwen3_5TextForCausalLM"]
    text_cfg_dict["model_type"] = "qwen3_5_text"
    text_cfg_dict["torch_dtype"] = "bfloat16"

    from transformers import AutoConfig as AC
    text_cfg = AC.for_model(**text_cfg_dict) if hasattr(AC, "for_model") else None
    if text_cfg is None:
        # construct via class
        from transformers import CONFIG_MAPPING
        cls = CONFIG_MAPPING["qwen3_5_text"]
        text_cfg = cls(**{k: v for k, v in text_cfg_dict.items()
                          if k in cls.__init__.__code__.co_varnames or True})
        for k, v in text_cfg_dict.items():
            try:
                setattr(text_cfg, k, v)
            except Exception:
                pass

    # Build wrapper
    from transformers import AutoModelForCausalLM as AML
    try:
        standalone = AML.from_config(text_cfg, trust_remote_code=True)
    except Exception:
        # Fallback: directly use the text submodule with lm_head from full
        standalone = text
        if hasattr(full, "lm_head"):
            standalone.lm_head = full.lm_head
        elif hasattr(full, "text_model") and hasattr(full, "lm_head"):
            standalone.lm_head = full.lm_head

    # Copy weights by name
    print("[extract] copying text weights...")
    src_state = text.state_dict()
    if hasattr(standalone, "state_dict"):
        dst_state = standalone.state_dict()
        new_state = {}
        for k, v in src_state.items():
            # try direct match
            matches = [dk for dk in dst_state if dk.endswith(k) or k.endswith(dk)]
            if matches:
                new_state[matches[0]] = v
            else:
                new_state[k] = v
        # also try lm_head from full
        if hasattr(full, "lm_head"):
            new_state["lm_head.weight"] = full.lm_head.weight.data
        missing, unexpected = standalone.load_state_dict(new_state, strict=False)
        print(f"[extract] load: missing={len(missing)} unexpected={len(unexpected)}")

    print(f"[extract] saving to {dst}...")
    standalone.save_pretrained(dst, safe_serialization=True)
    text_cfg.save_pretrained(dst)

    print("[extract] copying tokenizer + chat template...")
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                  "merges.txt", "chat_template.jinja", "generation_config.json",
                  "special_tokens_map.json"]:
        s = src / fname
        if s.exists():
            shutil.copy(s, dst / fname)

    print("[extract] done.")


if __name__ == "__main__":
    main()
