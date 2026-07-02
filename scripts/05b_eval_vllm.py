#!/usr/bin/env python3
"""Fast eval over 100 questions using vLLM server replicas.

Reads judge/eval data and posts to N vLLM endpoints in parallel, writes
one batch_<ts>.jsonl to judge/pending/ for Claude to grade.

Usage:
  python 05b_eval_vllm.py --model_path Qwen/Qwen3-8B \
      --model_label Qwen3-8B-base --served_name Qwen3-8B-base

Assumes vLLM servers (run_vllm_replicas.sh) already up. Endpoints read from
$VLLM_LOG_DIR/vllm_base_urls.txt or --urls.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVAL_DATA = ROOT / "data" / "processed" / "eval.jsonl"
JUDGE_DIR = ROOT / "judge"
PENDING_DIR = JUDGE_DIR / "pending"


def load_rows() -> List[Dict[str, Any]]:
    rows = []
    with EVAL_DATA.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def post_completion(url: str, model_name: str, prompt: str,
                    max_tokens: int, temperature: float, timeout: int = 600) -> str:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
    }
    r = requests.post(f"{url}/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    obj = r.json()
    return obj["choices"][0]["message"]["content"]


def apply_chat_local(tokenizer, q: str, enable_thinking) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except Exception:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_label", required=True)
    p.add_argument("--checkpoint_label", default="base")
    p.add_argument("--served_name", required=True, help="must match vLLM --served-model-name")
    p.add_argument("--urls", default=None, help="comma list; default reads $VLLM_LOG_DIR/vllm_base_urls.txt")
    p.add_argument("--chat_template_enable_thinking", default="auto")
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--tokenizer", default=None, help="for applying chat template; default = served_name")
    args = p.parse_args()

    # URLs
    if args.urls:
        urls = [u.strip().rstrip("/") for u in args.urls.split(",") if u.strip()]
    else:
        url_file = Path(os.environ.get("VLLM_LOG_DIR", "./logs/vllm")) / "vllm_base_urls.txt"
        if not url_file.exists():
            raise SystemExit(f"No URLs given and {url_file} missing")
        urls = [u.strip().rstrip("/") for u in url_file.read_text().split(",") if u.strip()]
    print(f"vLLM endpoints: {urls}")

    # Tokenizer for chat template
    tok_src = args.tokenizer or args.served_name
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

    enable_thinking = None
    if args.chat_template_enable_thinking == "true":
        enable_thinking = True
    elif args.chat_template_enable_thinking == "false":
        enable_thinking = False

    rows = load_rows()
    print(f"Loaded {len(rows)} eval questions")

    # Pre-render prompts
    prompts = []
    for r in rows:
        q = r["question"]
        prompts.append((r, apply_chat_local(tokenizer, q, enable_thinking)))

    # Round-robin assign URLs
    entries: List[Dict[str, Any]] = [None] * len(rows)
    t0 = time.time()

    def work(idx_url):
        idx, url = idx_url
        r, prompt = prompts[idx]
        try:
            text = post_completion(url, args.served_name, prompt,
                                   args.max_new_tokens, args.temperature)
        except Exception as e:
            text = f"[GEN_ERROR] {type(e).__name__}: {e}"
        return idx, r, text

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls) * 4) as ex:
        futures = [ex.submit(work, (i, urls[i % len(urls)])) for i in range(len(rows))]
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            idx, r, text = fut.result()
            entries[idx] = {
                "question": r["question"],
                "gold_answer": r.get("gold_answer", ""),
                "full_thinking": text,
                "candidate_answer": _extract_post_think(text),
                "category": r.get("category", ""),
                "doi": r.get("doi", ""),
                "title": r.get("title", ""),
                "source_text_excerpt": r.get("source_text", "")[:1500],
            }
            done += 1
            if done % 10 == 0:
                elapsed = time.time() - t0
                print(f"  generated {done}/{len(rows)}  elapsed={elapsed:.0f}s")

    # Write one pending batch for Claude judge
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    from src.judge_queue import JudgeQueue
    q = JudgeQueue(str(JUDGE_DIR))
    out = q.write_pending(
        entries, batch_kind="eval_metric",
        checkpoint=args.checkpoint_label, model=args.model_label,
    )
    print(f"\nWrote {len(entries)} eval entries -> {out}")
    print(f"Total elapsed: {time.time()-t0:.0f}s")
    print("Now: bash scripts/run_judge_worker.sh")


def _extract_post_think(text: str) -> str:
    if "</think>" not in text:
        return text.strip()
    idx = text.rfind("</think>")
    return text[idx + len("</think"):].strip()


if __name__ == "__main__":
    main()
