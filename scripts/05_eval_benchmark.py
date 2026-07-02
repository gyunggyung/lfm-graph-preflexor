#!/usr/bin/env python3
"""Evaluate a model on the 100-question Graph-PRefLexOR benchmark.

Paper §2.1 + §4.2: 3 metrics (Reasoning Quality, Intellectual Depth, Reasoning
Traceability), 0-10 scale, judged by Claude opus-4.7. We re-use the file-based
judge queue so Claude can grade offline.

Two-stage:
  Stage 1: this script generates model responses (vLLM) and writes pending eval
           entries to judge/pending/.
  Stage 2: claude evaluates by reading pending, writing done. (Separate worker
           invocation - see scripts/run_judge_worker.sh.)
  Stage 3: this script (called with --collect) reads done entries and produces
           aggregate metrics.

Usage:
  python 05_eval_benchmark.py --model_path ./checkpoints/grpo_lfm25 \
      --model_label LFM2.5-8B-A1B-Graph-GRPO --generate
  # ... run judge worker as Claude ...
  python 05_eval_benchmark.py --model_path ./checkpoints/grpo_lfm25 \
      --model_label LFM2.5-8B-A1B-Graph-GRPO --collect
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVAL_DATA = ROOT / "data" / "processed" / "eval.jsonl"
JUDGE_DIR = ROOT / "judge"
LOG_DIR = ROOT / "logs" / "eval"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_eval_rows() -> List[Dict[str, Any]]:
    if not EVAL_DATA.exists():
        raise SystemExit(f"{EVAL_DATA} missing. Run 02_prepare_data.py first.")
    rows = []
    with EVAL_DATA.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stage_generate(args) -> None:
    """Generate model outputs for all 100 questions, write to judge/pending."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.judge_queue import JudgeQueue

    rows = load_eval_rows()
    print(f"Loaded {len(rows)} benchmark questions")

    print(f"Loading model: {args.model_path}")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    enable_thinking = None
    if args.chat_template_enable_thinking == "true":
        enable_thinking = True
    elif args.chat_template_enable_thinking == "false":
        enable_thinking = False

    queue = JudgeQueue(args.judge_queue_dir or str(JUDGE_DIR))
    entries: List[Dict[str, Any]] = []

    for i, r in enumerate(rows):
        q = r["question"]
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except Exception:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True,
            )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=max(args.temperature, 1e-3),
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=False)

        entries.append({
            "question": q,
            "gold_answer": r.get("gold_answer", ""),
            "full_thinking": text,
            "candidate_answer": _extract_post_think(text),
            "category": r.get("category", ""),
            "doi": r.get("doi", ""),
            "title": r.get("title", ""),
            "source_text_excerpt": r.get("source_text", "")[:1500],
        })
        if (i + 1) % 10 == 0:
            print(f"  generated {i+1}/{len(rows)}")

    out_path = queue.write_pending(
        entries, batch_kind="eval_metric",
        checkpoint=args.checkpoint_label,
        model=args.model_label,
    )
    print(f"\nWrote {len(entries)} eval entries to {out_path}")
    print("Now invoke Claude judge worker (scripts/run_judge_worker.sh)")
    print("Then re-run with --collect to compute aggregate metrics.")


def _extract_post_think(text: str) -> str:
    if "</think>" not in text:
        return text.strip()
    idx = text.rfind("</think>")
    return text[idx + len("</think>"):].strip()


def stage_collect(args) -> None:
    """Read judge/done/ and produce aggregate metrics."""
    from src.judge_queue import JudgeQueue

    queue = JudgeQueue(args.judge_queue_dir or str(JUDGE_DIR))
    done_dir = Path(args.judge_queue_dir or JUDGE_DIR) / "done"
    if not done_dir.exists():
        raise SystemExit("No done directory. Run Claude judge worker first.")

    # Read all done entries (don't archive; eval is one-shot per checkpoint)
    rows: List[Dict[str, Any]] = []
    for p in sorted(done_dir.glob("batch_*.jsonl")):
        # Only consume entries with eval_metric type and matching model label
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("type") != "eval_metric":
                    continue
                if obj.get("model") != args.model_label:
                    continue
                rows.append(obj)

    if not rows:
        raise SystemExit(f"No eval judgments found for model_label={args.model_label!r}")

    metrics = ["reasoning_quality", "intellectual_depth", "reasoning_traceability"]
    sums = {m: 0.0 for m in metrics}
    n_per_metric = {m: 0 for m in metrics}
    for r in rows:
        s = r.get("scores", {})
        for m in metrics:
            v = s.get(m)
            if v is not None:
                sums[m] += float(v)
                n_per_metric[m] += 1

    avgs = {m: (sums[m] / n_per_metric[m] if n_per_metric[m] else 0.0) for m in metrics}
    overall = sum(avgs.values()) / max(1, len(metrics))

    out = {
        "model": args.model_label,
        "checkpoint": args.checkpoint_label,
        "n_evaluated": len(rows),
        "metrics": avgs,
        "overall_score": overall,
        "per_metric_count": n_per_metric,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))

    out_file = LOG_DIR / f"{args.model_label}_scores.json"
    out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_file}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_label", required=True)
    p.add_argument("--checkpoint_label", default="final")
    p.add_argument("--judge_queue_dir", default=None)
    p.add_argument("--chat_template_enable_thinking", default="auto")
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--generate", action="store_true")
    p.add_argument("--collect", action="store_true")
    args = p.parse_args()

    if not args.generate and not args.collect:
        raise SystemExit("Pass --generate (stage 1) or --collect (stage 3)")
    if args.generate:
        stage_generate(args)
    if args.collect:
        stage_collect(args)


if __name__ == "__main__":
    main()
