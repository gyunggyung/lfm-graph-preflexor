#!/usr/bin/env python3
"""Judge worker entry point: prints pending batches in a Claude-friendly format.

This script is invoked from scripts/run_judge_worker.sh. It:
  1. Lists all judge/pending/*.jsonl batches
  2. For each, reads the entries and prints them as numbered prompts
  3. Waits for Claude (the operator) to paste JSON results on stdin
  4. Validates and writes them to judge/done/<same_name>.jsonl

When invoked with --list-only, just prints the queue state without prompting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.judge_queue import JudgeQueue


def summarize_entry(e: Dict[str, Any], idx: int) -> str:
    """Render one pending entry for Claude to read."""
    etype = e.get("type", "unknown")
    out = []
    out.append(f"\n--- entry {idx} | id={e.get('id')} | type={etype} ---")
    out.append(f"checkpoint: {e.get('checkpoint')} | model: {e.get('model')}")
    out.append(f"QUESTION:\n{e.get('question','')}")
    if etype == "grpo_reward":
        out.append(f"\nGOLD ANSWER:\n{e.get('gold_answer','')[:1500]}")
        out.append(f"\nCANDIDATE ANSWER:\n{e.get('candidate_answer','')[:1500]}")
        if e.get("graph_json"):
            out.append(f"\nGRAPH_JSON:\n{e['graph_json'][:1500]}")
    else:  # eval_metric
        out.append(f"\nFULL THINKING/RESPONSE:\n{e.get('full_thinking','')[:4000]}")
        if e.get("gold_answer"):
            out.append(f"\nGOLD ANSWER (if any):\n{e['gold_answer'][:800]}")
    return "\n".join(out)


def render_result_template(entries: List[Dict[str, Any]]) -> str:
    """Print a JSON template that Claude should fill in."""
    template = []
    for e in entries:
        if e.get("type") == "grpo_reward":
            template.append({
                "id": e["id"],
                "scores": {
                    "correctness": 0.0,
                    "graph_utility": 0.0,
                },
                "justification": "",
            })
        else:
            template.append({
                "id": e["id"],
                "scores": {
                    "reasoning_quality": 0.0,
                    "intellectual_depth": 0.0,
                    "reasoning_traceability": 0.0,
                },
                "justification": "",
            })
    return json.dumps(template, indent=2, ensure_ascii=False)


def parse_results(text: str) -> List[Dict[str, Any]]:
    """Parse a JSON array (possibly with markdown fences) into result dicts."""
    text = text.strip()
    if text.startswith("```"):
        # strip ```json ... ``` fence
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    obj = json.loads(text)
    if not isinstance(obj, list):
        raise ValueError("Expected a JSON array of result objects")
    return obj


def validate(results: List[Dict[str, Any]], entries: List[Dict[str, Any]]) -> None:
    by_id = {e["id"]: e for e in entries}
    seen = set()
    for r in results:
        rid = r.get("id")
        if rid not in by_id:
            raise ValueError(f"Unknown id in results: {rid}")
        if rid in seen:
            raise ValueError(f"Duplicate id in results: {rid}")
        seen.add(rid)
        etype = by_id[rid].get("type")
        scores = r.get("scores", {})
        if not isinstance(scores, dict):
            raise ValueError(f"scores must be object for {rid}")
        if etype == "grpo_reward":
            for k in ("correctness", "graph_utility"):
                v = scores.get(k)
                if v is None or not (0.0 <= float(v) <= 1.0):
                    raise ValueError(f"{rid}.{k} must be in [0,1], got {v}")
        else:  # eval_metric
            for k in ("reasoning_quality", "intellectual_depth", "reasoning_traceability"):
                v = scores.get(k)
                if v is None or not (0.0 <= float(v) <= 10.0):
                    raise ValueError(f"{rid}.{k} must be in [0,10], got {v}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queue_dir", default=str(ROOT / "judge"))
    p.add_argument("--list-only", action="store_true",
                   help="Only list pending batches, don't enter interactive loop")
    p.add_argument("--batch", default=None,
                   help="Process only this batch_id (substring match on filename)")
    p.add_argument("--max-entries-per-batch", type=int, default=32)
    args = p.parse_args()

    q = JudgeQueue(args.queue_dir)
    pending = q.list_pending()
    if args.batch:
        pending = [p_ for p_ in pending if args.batch in p_.name]
    if not pending:
        print("No pending batches.")
        return

    print(f"Found {len(pending)} pending batch(es).")
    for p_ in pending:
        print(f"  {p_.name}")

    if args.list_only:
        return

    for batch_path in pending:
        entries = q.read_pending_batch(batch_path)
        if len(entries) > args.max_entries_per_batch:
            print(f"\n[skip] {batch_path.name}: {len(entries)} entries > --max-entries-per-batch={args.max_entries_per_batch}")
            print("       raise the limit or split the batch.")
            continue

        print(f"\n{'='*70}\nBatch: {batch_path.name}  ({len(entries)} entries)\n{'='*70}")
        for i, e in enumerate(entries):
            print(summarize_entry(e, i))

        print("\nFill in this JSON template. Each id must match an entry above.")
        print("For grpo_reward: correctness [0,1], graph_utility [0,1].")
        print("For eval_metric: 3 metrics, each [0,10].")
        print("\nTEMPLATE (paste with answers, wrap in ```json ... ``` if you like):")
        print(render_result_template(entries))

        print("\nPaste JSON results below, then a blank line and EOF (Ctrl-D):")
        chunks = []
        try:
            while True:
                line = input()
                if line.strip() == "EOF":
                    break
                chunks.append(line)
        except EOFError:
            pass

        text = "\n".join(chunks).strip()
        if not text:
            print("(empty input; skipping batch)")
            continue
        try:
            results = parse_results(text)
            validate(results, entries)
        except Exception as e:
            print(f"PARSE/VALIDATION ERROR: {e}")
            print("Batch left pending. Fix and re-run.")
            continue

        done_path = q.write_done(batch_path, results)
        print(f"Wrote {done_path}")


if __name__ == "__main__":
    main()
