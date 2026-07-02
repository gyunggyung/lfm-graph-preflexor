#!/usr/bin/env python3
"""Download Graph-PRefLexOR datasets from HuggingFace.

Outputs:
  data/raw/train_10K.jsonl         (lamm-mit/graph_reasoning_10K)
  data/raw/benchmark_100.jsonl     (lamm-mit/graph-preflexor-grpo-benchmark)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    env_path = ROOT.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("export HF_TOKEN="):
                HF_TOKEN = line.split("=", 1)[1].strip().strip('"').strip("'")
                os.environ["HF_TOKEN"] = HF_TOKEN
                break

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not found in env or ../.env", file=sys.stderr)


def to_jsonl(ds, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for row in ds:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    from datasets import load_dataset

    train_repo = "lamm-mit/graph_reasoning_10K"
    bench_repo = "lamm-mit/graph-preflexor-grpo-benchmark"

    print(f"[1/2] Downloading {train_repo} ...")
    train_ds = load_dataset(train_repo, split="train", token=HF_TOKEN)
    n_train = to_jsonl(train_ds, RAW_DIR / "train_10K.jsonl")
    print(f"  → {RAW_DIR / 'train_10K.jsonl'} ({n_train} rows)")
    print(f"  columns: {train_ds.column_names}")

    print(f"[2/2] Downloading {bench_repo} ...")
    bench_ds = load_dataset(bench_repo, split="train", token=HF_TOKEN)
    n_bench = to_jsonl(bench_ds, RAW_DIR / "benchmark_100.jsonl")
    print(f"  → {RAW_DIR / 'benchmark_100.jsonl'} ({n_bench} rows)")
    print(f"  columns: {bench_ds.column_names}")

    print(f"\nDone. {n_train} train + {n_bench} benchmark rows.")


if __name__ == "__main__":
    main()
