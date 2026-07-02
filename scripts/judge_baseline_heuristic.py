#!/usr/bin/env python3
"""Heuristic baseline judge for un-trained models.

For base Qwen3-8B (or any model that hasn't been ORPO'd to produce the
5-sentinel format), we apply a fast rubric:

  reasoning_quality [0-10]: based on length, structure (bullets/sections),
                            and presence of mechanistic keywords.
  intellectual_depth [0-10]: based on cross-domain term overlap, length,
                             density of novel concepts.
  reasoning_traceability [0-10]: 0 unless the model emits <graph_json>;
                                 otherwise based on graph node/edge count
                                 and whether synthesis references them.

This is NOT a substitute for Claude-as-judge on trained models. It's a
fast baseline for the un-trained checkpoint so we have a number to compare
against.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.judge_queue import JudgeQueue


GRAPH_JSON_RE = re.compile(r"<graph_json>\s*(\{.*?\})\s*</graph_json>", re.DOTALL)
SENTINELS = ["<brainstorm>", "<graph>", "<patterns>", "<synthesis>", "<think>"]

CROSS_DOMAIN_KEYWORDS = [
    "multi-scale", "cross-domain", "trade-off", "tradeoff", "non-monotonic",
    "hidden variable", "emergent", "feedback loop", "coupling", "hierarchy",
    "boundary", "scale-dependent", "stochastic", "nonlinear", "distributed",
    "heterogeneous", "multi-physics", "open-ended", "adaptive",
]

MECHANISTIC_KEYWORDS = [
    "because", "therefore", "thus", "leads to", "driven by", "results in",
    "mechanism", "pathway", "process", "transforms", "regulates", "mediates",
    "via", "through", "causes", "governs", "underlies",
]


def count_keywords(text: str, words: List[str]) -> int:
    t = text.lower()
    return sum(t.count(w) for w in words)


def extract_post_think(text: str) -> str:
    if "</think>" not in text:
        return text.strip()
    return text[text.rfind("</think>") + len("</think>"):].strip()


def extract_graph(full_text: str) -> Dict[str, Any] | None:
    m = GRAPH_JSON_RE.search(full_text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def score_reasoning_quality(answer: str, full: str) -> float:
    if not answer or len(answer) < 100:
        return 1.5
    score = 2.0
    # length-based floor (up to 4)
    L = min(len(answer), 4000)
    score += (L / 4000) * 2.0
    # structural markers
    if re.search(r"(?m)^\s*[-*]\s", answer):
        score += 0.7  # bullet points
    n = answer.count("\n## ") + answer.count("\n### ")
    score += min(n * 0.3, 1.0)
    # mechanistic reasoning
    m = count_keywords(answer, MECHANISTIC_KEYWORDS)
    score += min(m * 0.1, 1.5)
    # penalize pure repetition
    unique_words = len(set(answer.lower().split()))
    total = max(len(answer.split()), 1)
    ttr = unique_words / total
    if ttr < 0.25:
        score -= 1.0
    return max(0.0, min(10.0, score))


def score_intellectual_depth(answer: str, full: str) -> float:
    if not answer:
        return 0.5
    score = 1.0
    L = min(len(answer), 4000)
    score += (L / 4000) * 1.5
    cd = count_keywords(answer + " " + full, CROSS_DOMAIN_KEYWORDS)
    score += min(cd * 0.5, 3.5)
    # concept density (rough)
    words = answer.split()
    if words:
        unique_ratio = len(set(w.lower() for w in words)) / max(len(words), 1)
        score += unique_ratio * 2.0
    return max(0.0, min(10.0, score))


def score_reasoning_traceability(full: str, answer: str) -> float:
    # 0 if no graph_json tag at all (this is the main delta the paper measures)
    if "<graph_json>" not in full:
        return 1.0
    score = 3.0
    g = extract_graph(full)
    if g:
        nodes = g.get("nodes", []) or []
        edges = g.get("edges", []) or []
        score += min(len(nodes) * 0.2, 2.0)
        score += min(len(edges) * 0.2, 2.0)
    # synthesis referencing graph concepts
    syn_match = re.search(r"<synthesis>(.*?)</synthesis>", full, re.DOTALL)
    if syn_match and len(syn_match.group(1)) > 100:
        score += 1.5
    # patterns block present
    if "<patterns>" in full and "</patterns>" in full:
        score += 1.0
    return max(0.0, min(10.0, score))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch_glob", default="batch_*.jsonl",
                   help="in judge/pending/, which batch file to process")
    p.add_argument("--judge_dir", default=str(ROOT / "judge"))
    p.add_argument("--model_filter", default=None,
                   help="only judge entries with this model field")
    args = p.parse_args()

    q = JudgeQueue(args.judge_dir)
    pending = q.list_pending()
    if args.batch_glob != "batch_*.jsonl":
        pending = [p_ for p_ in pending if args.batch_glob in p_.name]

    for batch_path in pending:
        entries = q.read_pending_batch(batch_path)
        if args.model_filter:
            entries = [e for e in entries if e.get("model") == args.model_filter]
        if not entries:
            continue
        print(f"Judging {len(entries)} entries from {batch_path.name}")

        results = []
        for e in entries:
            full = e.get("full_thinking", "") or ""
            answer = e.get("candidate_answer", "") or extract_post_think(full)
            rq = score_reasoning_quality(answer, full)
            idd = score_intellectual_depth(answer, full)
            rt = score_reasoning_traceability(full, answer)
            results.append({
                "id": e["id"],
                "scores": {
                    "reasoning_quality": round(rq, 2),
                    "intellectual_depth": round(idd, 2),
                    "reasoning_traceability": round(rt, 2),
                },
                "justification": (
                    f"heuristic: rq(len={len(answer)},mech_kw={count_keywords(answer, MECHANISTIC_KEYWORDS)}); "
                    f"depth(cross_kw={count_keywords(full, CROSS_DOMAIN_KEYWORDS)}); "
                    f"trace({'graph' if '<graph_json>' in full else 'no_graph'})"
                ),
            })

        done_path = q.write_done(batch_path, results)
        print(f"  Wrote {done_path}")


if __name__ == "__main__":
    main()
