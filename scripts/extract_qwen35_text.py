#!/usr/bin/env python3
"""Convert Qwen3.5-9B (multimodal) to standalone text-only CausalLM.

Strategy: rename safetensors keys (model.language_model.X -> model.X) and
write a flat qwen3_5_text config. AutoModelForCausalLM should pick up
Qwen3_5TextForCausalLM via the qwen3_5_text model_type.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    src = Path(a.src); dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # 1) Write flat text-only config from text_config sub-dict
    with open(src / "config.json") as f:
        full_cfg = json.load(f)
    text_cfg = dict(full_cfg.get("text_config", {}))
    text_cfg["architectures"] = ["Qwen3_5TextForCausalLM"]
    text_cfg["model_type"] = "qwen3_5_text"
    text_cfg["torch_dtype"] = "bfloat16"
    text_cfg["transformers_version"] = full_cfg.get("transformers_version", "5.5.4")
    with open(dst / "config.json", "w") as f:
        json.dump(text_cfg, f, indent=2)
    print(f"[extract] wrote text config: {dst / 'config.json'}")

    # 2) Re-key weights: model.language_model.X -> model.X (drop visual)
    shards = sorted([f for f in os.listdir(src) if f.endswith(".safetensors")])
    total_renamed = 0
    total_unchanged = 0
    total_dropped = 0
    for shard in shards:
        sd = load_file(str(src / shard))
        new_sd = {}
        for k, v in sd.items():
            if "visual" in k or "vision" in k:
                total_dropped += 1
                continue
            if k.startswith("model.language_model."):
                new_sd["model." + k[len("model.language_model."):]] = v
                total_renamed += 1
            else:
                new_sd[k] = v
                total_unchanged += 1
        out_name = shard
        save_file(new_sd, str(dst / out_name), metadata={"format": "pt"})
        del sd, new_sd
    print(f"[extract] renamed {total_renamed}, kept {total_unchanged}, dropped {total_dropped} (visual)")

    # 3) Copy tokenizer + chat template
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                  "merges.txt", "chat_template.jinja", "generation_config.json",
                  "special_tokens_map.json"]:
        s = src / fname
        if s.exists():
            shutil.copy(s, dst / fname)
            print(f"[extract] copied {fname}")

    print(f"[extract] done. -> {dst}")


if __name__ == "__main__":
    main()
