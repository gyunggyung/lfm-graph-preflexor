#!/usr/bin/env python3
"""4-GPU parallel eval via transformers. Each GPU loads the model once and
processes a shard of the 100 questions.

Workaround for vLLM's broken Qwen3.5 text-only support.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVAL_DATA = ROOT / "data" / "processed" / "eval.jsonl"
JUDGE_DIR = ROOT / "judge"
PENDING_DIR = JUDGE_DIR / "pending"


def load_rows():
    rows = []
    with EVAL_DATA.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_post_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def worker(shard_id: int, gpu_id: int, model_path: str, items: list,
           enable_thinking, max_new_tokens: int, temperature: float,
           served_name: str, gpu_base: int) -> list:
    real_gpu = gpu_base + shard_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(real_gpu)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[shard {shard_id}] gpu={real_gpu} loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    print(f"[shard {shard_id}] model loaded, generating {len(items)}", flush=True)

    out = []
    t0 = time.time()
    for i, r in enumerate(items):
        q = r["question"]
        try:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except Exception:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True,
            )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            do_sample = temperature > 0
            ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 0.01) if do_sample else 1.0,
                top_p=0.95 if do_sample else 1.0,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(ids[0, inputs.input_ids.shape[1]:],
                          skip_special_tokens=False)
        out.append({
            "question": r["question"],
            "gold_answer": r.get("gold_answer", ""),
            "full_thinking": text,
            "candidate_answer": _extract_post_think(text),
            "category": r.get("category", ""),
            "doi": r.get("doi", ""),
            "title": r.get("title", ""),
            "source_text_excerpt": r.get("source_text", "")[:1500],
        })
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"[shard {shard_id}] {i+1}/{len(items)} elapsed={elapsed:.0f}s",
                  flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_label", required=True)
    p.add_argument("--checkpoint_label", default="orpo")
    p.add_argument("--num_gpus", type=int, default=4)
    p.add_argument("--gpu_base", type=int, default=2,
                   help="first GPU id to use (default 2 → GPUs 2,3,4,5)")
    p.add_argument("--chat_template_enable_thinking", default="auto")
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.2)
    args = p.parse_args()

    rows = load_rows()
    print(f"Loaded {len(rows)} eval questions")

    enable_thinking = None
    if args.chat_template_enable_thinking == "true":
        enable_thinking = True
    elif args.chat_template_enable_thinking == "false":
        enable_thinking = False

    # Shard rows across GPUs
    shards = [[] for _ in range(args.num_gpus)]
    for i, r in enumerate(rows):
        shards[i % args.num_gpus].append(r)

    t0 = time.time()
    all_results = [None] * len(rows)
    with ProcessPoolExecutor(max_workers=args.num_gpus) as ex:
        futs = {}
        for shard_id, items in enumerate(shards):
            if not items:
                continue
            # Map shard results back to global indices
            global_idxs = list(range(shard_id, len(rows), args.num_gpus))
            fut = ex.submit(worker, shard_id, shard_id, args.model_path,
                            items, enable_thinking, args.max_new_tokens,
                            args.temperature, args.model_label, args.gpu_base)
            futs[fut] = global_idxs

        done = 0
        for fut in as_completed(futs):
            results = fut.result()
            gidxs = futs[fut]
            for r, gidx in zip(results, gidxs):
                all_results[gidx] = r
            done += len(results)
            print(f"  shard done: {done}/{len(rows)} cumulative")

    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    from src.judge_queue import JudgeQueue
    q = JudgeQueue(str(JUDGE_DIR))
    out = q.write_pending(
        all_results, batch_kind="eval_metric",
        checkpoint=args.checkpoint_label, model=args.model_label,
    )
    print(f"\nWrote {len(all_results)} eval entries -> {out}")
    print(f"Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
