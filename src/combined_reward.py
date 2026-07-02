"""Combined reward for Graph-GRPO with no external API.

Total reward = sum(weight_k * score_k) over 6 components (paper Eq. (4)):
  correctness       (0.30)   embedding-based (this repo)
  format            (0.15)   programmatic
  graph_utility     (0.25)   embedding-based (this repo)
  graph_networkx    (0.10)   programmatic
  graph_diversity   (0.10)   programmatic (sentence-BERT)
  graph_structure   (0.10)   programmatic (networkx)

Optionally, when a Claude-judged result is available for a given completion id
(from the judge queue), we *blend* the embedding reward with Claude's score:
  blended = alpha * embedding + (1 - alpha) * claude
where alpha defaults to 0.5. This is the calibration hook: as Claude judgments
accumulate, the trainer can decide whether embedding reward is well-calibrated.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from . import embedding_rewards as er
from . import programmatic_rewards as pr
from .judge_queue import JudgeQueue

logger = logging.getLogger("graph_reward")


@dataclass
class RewardWeights:
    correctness: float = 0.30
    format: float = 0.15
    graph_utility: float = 0.25
    graph_networkx: float = 0.10
    graph_diversity: float = 0.10
    graph_structure: float = 0.10

    def as_dict(self) -> Dict[str, float]:
        return {
            "correctness": self.correctness,
            "format": self.format,
            "graph_utility": self.graph_utility,
            "graph_networkx": self.graph_networkx,
            "graph_diversity": self.graph_diversity,
            "graph_structure": self.graph_structure,
        }

    def sum(self) -> float:
        return sum(self.as_dict().values())


@dataclass
class RewardBreakdown:
    correctness: float = 0.0
    format: float = 0.0
    graph_utility: float = 0.0
    graph_networkx: float = 0.0
    graph_diversity: float = 0.0
    graph_structure: float = 0.0
    total: float = 0.0
    used_claude_blend: bool = False

    def as_dict(self) -> Dict[str, float]:
        return {
            "correctness": self.correctness,
            "format": self.format,
            "graph_utility": self.graph_utility,
            "graph_networkx": self.graph_networkx,
            "graph_diversity": self.graph_diversity,
            "graph_structure": self.graph_structure,
            "total": self.total,
            "used_claude_blend": float(self.used_claude_blend),
        }


class CombinedReward:
    """Stateful reward computer: lazily inits embedder, owns the judge queue."""

    def __init__(
        self,
        weights: RewardWeights,
        embed_model: str = "BAAI/bge-base-en-v1.5",
        judge_queue_dir: Optional[str] = None,
        claude_blend_alpha: float = 0.5,
    ):
        wsum = weights.sum()
        if abs(wsum - 1.0) > 0.05:
            logger.warning("Reward weights sum to %.3f, expected 1.0", wsum)
        self.weights = weights
        self.embed = er.Embedder(model_name=embed_model)
        self.claude_blend_alpha = claude_blend_alpha
        self.queue = JudgeQueue(judge_queue_dir) if judge_queue_dir else None
        # id -> {"correctness": float, "graph_utility": float}
        self._claude_cache: Dict[str, Dict[str, float]] = {}
        self._cache_lock = threading.Lock()

    def refresh_claude_cache(self) -> int:
        """Pull all judged entries from queue.done into the cache. Returns count."""
        if self.queue is None:
            return 0
        new = self.queue.drain_done()
        with self._cache_lock:
            for r in new:
                rid = r.get("id")
                scores = r.get("scores", {})
                if rid and scores:
                    entry = {}
                    if "correctness" in scores:
                        entry["correctness"] = float(scores["correctness"])
                    if "graph_utility" in scores:
                        entry["graph_utility"] = float(scores["graph_utility"])
                    if entry:
                        self._claude_cache[rid] = entry
        return len(new)

    def submit_pending(
        self,
        completions: List[Dict[str, Any]],
        checkpoint: str,
        model: str,
    ) -> Optional[str]:
        """Queue completions for Claude judgment (non-blocking). Returns batch_id or None."""
        if self.queue is None or not completions:
            return None
        entries = []
        for c in completions:
            entries.append({
                "question": c["question"],
                "gold_answer": c["gold_answer"],
                "candidate_answer": c["candidate_answer"],
                "graph_json": c.get("graph_json") or "",
                "completion_id": c["id"],
            })
        path = self.queue.write_pending(
            entries, batch_kind="grpo_reward", checkpoint=checkpoint, model=model,
        )
        return path.stem.replace("batch_", "")

    def compute(
        self,
        question: str,
        gold_answer: str,
        full_output: str,
        completion_id: Optional[str] = None,
    ) -> RewardBreakdown:
        # Format / programmatic
        fmt = pr.score_format(full_output)
        gnx = pr.score_graph_networkx(full_output)
        gdiv = pr.score_graph_diversity(full_output)
        gstruct = pr.score_graph_structure(full_output)

        # Embedding-based
        candidate_answer = pr.extract_post_think(full_output)
        if not candidate_answer:
            candidate_answer = full_output[-1500:]
        corr = er.score_correctness_embedding(self.embed, candidate_answer, gold_answer)
        gj_str = pr.extract_graph_json_str(full_output)
        gut = er.score_graph_utility_embedding(self.embed, gj_str, gold_answer)

        used_blend = False
        if completion_id is not None:
            with self._cache_lock:
                claude = self._claude_cache.get(completion_id)
            if claude:
                a = self.claude_blend_alpha
                corr = a * corr + (1 - a) * claude.get("correctness", corr)
                gut = a * gut + (1 - a) * claude.get("graph_utility", gut)
                used_blend = True

        w = self.weights
        total = (
            w.correctness * corr
            + w.format * fmt
            + w.graph_utility * gut
            + w.graph_networkx * gnx
            + w.graph_diversity * gdiv
            + w.graph_structure * gstruct
        )

        return RewardBreakdown(
            correctness=corr,
            format=fmt,
            graph_utility=gut,
            graph_networkx=gnx,
            graph_diversity=gdiv,
            graph_structure=gstruct,
            total=total,
            used_claude_blend=used_blend,
        )

    def compute_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[RewardBreakdown]:
        """Compute rewards for a list of dicts with keys:
        question, gold_answer, full_output, id (optional)."""
        if not items:
            return []
        # Batched embedding scores
        cand_pairs = []
        graph_pairs = []
        for it in items:
            full = it["full_output"]
            cand = pr.extract_post_think(full) or full[-1500:]
            cand_pairs.append((cand, it["gold_answer"]))
            graph_pairs.append((pr.extract_graph_json_str(full), it["gold_answer"]))
        corr_arr = er.batch_correctness(self.embed, cand_pairs)
        gut_arr = er.batch_graph_utility(self.embed, graph_pairs)

        out: List[RewardBreakdown] = []
        for i, it in enumerate(items):
            full = it["full_output"]
            fmt = pr.score_format(full)
            gnx = pr.score_graph_networkx(full)
            gdiv = pr.score_graph_diversity(full)
            gstruct = pr.score_graph_structure(full)
            corr = corr_arr[i]
            gut = gut_arr[i]
            used_blend = False

            cid = it.get("id")
            if cid is not None:
                with self._cache_lock:
                    claude = self._claude_cache.get(cid)
                if claude:
                    a = self.claude_blend_alpha
                    corr = a * corr + (1 - a) * claude.get("correctness", corr)
                    gut = a * gut + (1 - a) * claude.get("graph_utility", gut)
                    used_blend = True

            w = self.weights
            total = (
                w.correctness * corr
                + w.format * fmt
                + w.graph_utility * gut
                + w.graph_networkx * gnx
                + w.graph_diversity * gdiv
                + w.graph_structure * gstruct
            )
            out.append(RewardBreakdown(
                correctness=corr, format=fmt, graph_utility=gut,
                graph_networkx=gnx, graph_diversity=gdiv, graph_structure=gstruct,
                total=total, used_claude_blend=used_blend,
            ))
        return out
