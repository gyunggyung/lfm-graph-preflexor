#!/usr/bin/env python3
"""Prepare datasets for ORPO, GRPO, and eval stages.

Inputs (from 01_download_data.py):
  data/raw/train_10K.jsonl
  data/raw/benchmark_100.jsonl

Outputs:
  data/processed/orpo.jsonl       columns: prompt, chosen, rejected
  data/processed/grpo.jsonl       columns: prompt, answer, question
  data/processed/eval.jsonl       columns: question, gold_answer (or empty), category, doi, title

The upstream schema for graph_reasoning_10K is:
  prompt:   str             user question
  chosen:   str             full assistant completion with <brainstorm>...<synthesis> + answer
  rejected: str             shallow 1-3 sentence answer
  answer:   str (optional)  gold post-`</think>` answer (we extract if missing)

We keep the schema minimal and let the trainers apply chat templates.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


THINK_END = "</think>"


def extract_post_think_answer(chosen: str) -> str:
    """Extract text after the last </think> as the gold answer.

    Mirrors src/run_grpo_graph.py:extract_post_thinking_answer.
    """
    if THINK_END not in chosen:
        return chosen.strip()
    idx = chosen.rfind(THINK_END)
    return chosen[idx + len(THINK_END):].strip()


def has_valid_graph_json(chosen: str) -> bool:
    """Discard rows whose graph_json fails to parse (paper §4.1.1)."""
    m = re.search(r"<graph_json>\s*(\{.*?\})\s*</graph_json>", chosen, flags=re.DOTALL)
    if not m:
        return False
    try:
        obj = json.loads(m.group(1))
        return isinstance(obj, dict) and "nodes" in obj
    except Exception:
        return False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def prepare_orpo(train_rows: list[dict[str, Any]], out_path: Path) -> int:
    """ORPO needs prompt/chosen/rejected."""
    kept = []
    dropped = 0
    for r in train_rows:
        prompt = (r.get("prompt") or "").strip()
        chosen = (r.get("chosen") or "").strip()
        rejected = (r.get("rejected") or "").strip()
        if not prompt or not chosen or not rejected:
            dropped += 1
            continue
        if not has_valid_graph_json(chosen):
            dropped += 1
            continue
        kept.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    n = write_jsonl(kept, out_path)
    print(f"  ORPO: kept {n}, dropped {dropped} (missing cols or invalid graph_json)")
    return n


def prepare_grpo(train_rows: list[dict[str, Any]], out_path: Path) -> int:
    """GRPO needs prompt (raw question) + answer (gold post-`</think>`).

    The trainer applies the chat template. We expose:
      prompt   = raw question (trainer applies chat template)
      answer   = gold answer (for reward scoring)
      question = same as prompt (passed through to reward fn)
    """
    kept = []
    dropped = 0
    for r in train_rows:
        prompt = (r.get("prompt") or "").strip()
        chosen = (r.get("chosen") or "").strip()
        if not prompt or not chosen:
            dropped += 1
            continue
        if not has_valid_graph_json(chosen):
            dropped += 1
            continue
        answer = (r.get("answer") or "").strip() or extract_post_think_answer(chosen)
        if not answer:
            dropped += 1
            continue
        kept.append({"prompt": prompt, "answer": answer, "question": prompt})
    n = write_jsonl(kept, out_path)
    print(f"  GRPO: kept {n}, dropped {dropped}")
    return n


def prepare_eval(bench_rows: list[dict[str, Any]], out_path: Path) -> int:
    """Eval: keep question + any gold reference + metadata.

    The benchmark is open-ended (paper §4.2). Some rows may not have a single gold
    answer; in that case `gold_answer` is empty and the eval judge grades against
    its own rubric using the question + paper context.
    """
    kept = []
    for r in bench_rows:
        question = (r.get("question") or r.get("prompt") or "").strip()
        if not question:
            continue
        kept.append({
            "question": question,
            "gold_answer": (r.get("gold_answer") or r.get("answer") or "").strip(),
            "category": (r.get("category") or r.get("reasoning_category")
                         or r.get("question_type") or ""),
            "doi": r.get("doi") or "",
            "title": (r.get("title") or r.get("paper_title") or "").strip(),
            "source_text": (r.get("source_text") or "")[:4000],
        })
    n = write_jsonl(kept, out_path)
    print(f"  Eval: kept {n} benchmark questions")
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="", help="Append suffix to output filenames (e.g. _lfm25)")
    args = p.parse_args()

    train_path = RAW_DIR / "train_10K.jsonl"
    bench_path = RAW_DIR / "benchmark_100.jsonl"

    if not train_path.exists() or not bench_path.exists():
        print(f"ERROR: raw data not found. Run 01_download_data.py first.", file=sys.stderr)
        sys.exit(1)

    train_rows = load_jsonl(train_path)
    bench_rows = load_jsonl(bench_path)
    print(f"Loaded {len(train_rows)} train + {len(bench_rows)} benchmark rows from raw/")

    sfx = args.suffix
    print("\nPreparing processed splits:")
    prepare_orpo(train_rows, PROC_DIR / f"orpo{sfx}.jsonl")
    prepare_grpo(train_rows, PROC_DIR / f"grpo{sfx}.jsonl")
    prepare_eval(bench_rows, PROC_DIR / f"eval{sfx}.jsonl")
    print("\nDone.")


if __name__ == "__main__":
    main()
