#!/usr/bin/env python3
"""Upload a trained adapter + eval JSON to the LLM-OS-Models HF org.

Reads HF_TOKEN from ../.env if not in env. Pushes:
  - adapter weights + tokenizer + a model card stub
  - logs/eval/<model_label>_scores.json as a sibling dataset (optional)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_token() -> str:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    env_path = ROOT.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("export HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("HF_TOKEN not found in env or ../.env")


def write_model_card(out_path: Path, hub_model_id: str, base_model: str, mode: str,
                     eval_path: Path | None) -> None:
    eval_section = ""
    if eval_path and eval_path.exists():
        try:
            data = json.loads(eval_path.read_text())
            metrics = data.get("metrics", {})
            overall = data.get("overall_score", 0.0)
            eval_section = (
                f"\n## Evaluation (100-question Graph-PRefLexOR benchmark)\n\n"
                f"- Overall: **{overall:.2f}** (mean of 3 metrics, 0-10 scale)\n"
                f"- Reasoning Quality: {metrics.get('reasoning_quality', 0):.2f}\n"
                f"- Intellectual Depth: {metrics.get('intellectual_depth', 0):.2f}\n"
                f"- Reasoning Traceability: {metrics.get('reasoning_traceability', 0):.2f}\n"
            )
        except Exception as e:
            eval_section = f"\n(eval parse failed: {e})\n"

    card = f"""---
license: apache-2.0
base_model: {base_model}
tags:
  - graph-preflexor
  - grpo
  - reasoning
  - lora
language:
  - en
pipeline_tag: text-generation
---

# {hub_model_id}

Adapter produced by `graph-preflexor-lfm25` (fork of `lamm-mit/graph-preflexor-grpo`,
arXiv 2607.00924v1). {mode} checkpoint.

- Base model: `{base_model}`
- Reward: 6-component (paper Eq. 4). Correctness + graph_utility use BGE embedding
  cosine instead of LLM-judge API, calibrated by Claude-as-judge via a file queue.

{eval_section}

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}", dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "{hub_model_id}")
tok = AutoTokenizer.from_pretrained("{hub_model_id}")
```

The model emits a structured reasoning trace `<brainstorm>...<graph>...<graph_json>...<patterns>...<synthesis>` inside `<think>`, then the final answer.
"""
    (out_path).write_text(card)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--local_dir", required=True)
    p.add_argument("--hub_model_id", required=True)
    p.add_argument("--base_model", required=True)
    p.add_argument("--mode", choices=["orpo", "grpo"], default="grpo")
    p.add_argument("--eval_json", default=None, help="Optional logs/eval/<label>_scores.json")
    p.add_argument("--public", action="store_true")
    args = p.parse_args()

    token = load_token()
    local = Path(args.local_dir).resolve()
    if not local.exists():
        raise SystemExit(f"{local} does not exist")

    eval_path = Path(args.eval_json).resolve() if args.eval_json else None

    # Write README.md
    write_model_card(local / "README.md", args.hub_model_id, args.base_model, args.mode, eval_path)

    # Push via huggingface_hub
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=args.hub_model_id, exist_ok=True,
                    private=(not args.public), repo_type="model")
    api.upload_folder(
        folder_path=str(local),
        repo_id=args.hub_model_id,
        repo_type="model",
        commit_message=f"Upload {args.mode} adapter ({local.name})",
    )

    # Optionally push eval JSON as a sibling dataset
    if eval_path and eval_path.exists():
        ds_repo = args.hub_model_id + "-eval"
        api.create_repo(repo_id=ds_repo, exist_ok=True,
                        private=(not args.public), repo_type="dataset")
        api.upload_file(
            path_or_fileobj=str(eval_path),
            path_in_repo=eval_path.name,
            repo_id=ds_repo,
            repo_type="dataset",
        )
        print(f"Eval JSON pushed to dataset {ds_repo}")

    print(f"Done. Model: https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
