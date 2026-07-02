"""File-based judge queue: training process writes pending batches, Claude reads them.

Layout under JUDGE_QUEUE_DIR (default ./judge):
  pending/batch_<ts>_<pid>_<seq>.jsonl     # training writes
  done/batch_<ts>_<pid>_<seq>.jsonl        # Claude writes (same id schema)
  archive/...                                # optional archive after ingest

Entry schema (pending):
  {
    "id": "<batch_id>|<idx>",
    "type": "grpo_reward" | "eval_metric",
    "batch_id": "<batch_id>",
    "idx": <int>,
    "issued_at": <iso>,
    "checkpoint": "step_<N>" | "final",
    "model": "LFM2.5-8B-A1B" | "Qwen3-8B" | ...,
    "question": "...",
    "gold_answer": "...",
    "candidate_answer": "...",
    "graph_json": "...",                    # for grpo_reward only
    "full_thinking": "...",                 # for eval_metric only
  }

Result schema (done):
  {
    "id": "<batch_id>|<idx>",
    "batch_id": "<batch_id>",
    "scores": {
      "correctness": <float>,                # for grpo_reward
      "graph_utility": <float>,              # for grpo_reward
      "reasoning_quality": <float>,          # for eval_metric (0-10)
      "intellectual_depth": <float>,         # for eval_metric
      "reasoning_traceability": <float>,     # for eval_metric
    },
    "justification": "<text>",
    "judged_at": <iso>,
    "judge": "claude-opus-4.7-role"
  }

The trainer's reward function uses embedding scores for non-blocking GRPO, then
periodically pulls done entries and:
  1. logs a calibration diff (embedding vs Claude)
  2. optionally nudges the embedding reward with a learned scalar
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class JudgeQueue:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "pending").mkdir(parents=True, exist_ok=True)
        (self.root / "done").mkdir(parents=True, exist_ok=True)
        (self.root / "archive").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = 0

    # ---------- producer (training side) ----------

    def write_pending(
        self,
        entries: List[Dict[str, Any]],
        batch_kind: str = "grpo_reward",
        checkpoint: str = "unknown",
        model: str = "unknown",
    ) -> Path:
        """Atomically write a batch of pending entries. Returns the file path."""
        if not entries:
            raise ValueError("write_pending: empty entries list")
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        with self._lock:
            self._seq += 1
            seq = f"{self._seq:04d}"
        batch_id = f"{ts}_{os.getpid()}_{seq}"
        path = self.root / "pending" / f"batch_{batch_id}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        issued = _now_iso()
        with tmp.open("w") as f:
            for i, e in enumerate(entries):
                row = {
                    "id": f"{batch_id}|{i}",
                    "type": batch_kind,
                    "batch_id": batch_id,
                    "idx": i,
                    "issued_at": issued,
                    "checkpoint": checkpoint,
                    "model": model,
                    **e,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        return path

    # ---------- consumer (Claude side) ----------

    def list_pending(self) -> List[Path]:
        return sorted((self.root / "pending").glob("batch_*.jsonl"))

    def read_pending_batch(self, path: Path) -> List[Dict[str, Any]]:
        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def write_done(self, path: Path, results: List[Dict[str, Any]]) -> Path:
        """Write results for one batch. `path` is the pending batch path."""
        done_path = self.root / "done" / path.name
        tmp = done_path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f:
            for r in results:
                r = {**r, "judged_at": _now_iso(), "judge": "claude-opus-4.7-role"}
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, done_path)
        # archive pending
        try:
            shutil.move(str(path), str(self.root / "archive" / path.name))
        except Exception:
            path.unlink(missing_ok=True)
        return done_path

    # ---------- trainer-side ingestion ----------

    def drain_done(self, since_batches: Optional[set] = None) -> List[Dict[str, Any]]:
        """Read all done entries and remove the files. Optionally only batches not
        already seen (caller passes `since_batches` set of batch_ids)."""
        out: List[Dict[str, Any]] = []
        for p in sorted((self.root / "done").glob("batch_*.jsonl")):
            batch_id = p.stem.replace("batch_", "")
            if since_batches is not None and batch_id in since_batches:
                continue
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
            # archive after read
            try:
                shutil.move(str(p), str(self.root / "archive" / p.name))
            except Exception:
                p.unlink(missing_ok=True)
        return out

    def calibration_diff(
        self,
        embedding_scores: Dict[str, Dict[str, float]],
        claude_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute mean abs diff between embedding and Claude scores per axis.

        embedding_scores: {id: {"correctness": x, "graph_utility": y, ...}}
        claude_results:   [{"id": ..., "scores": {...}}, ...]
        """
        diffs = {"correctness": [], "graph_utility": []}
        for r in claude_results:
            rid = r["id"]
            cs = r.get("scores", {})
            es = embedding_scores.get(rid, {})
            for axis in diffs:
                if axis in cs and axis in es:
                    diffs[axis].append(abs(float(cs[axis]) - float(es[axis])))
        out = {
            "n": len(claude_results),
            "mean_abs_diff": {
                k: (sum(v) / len(v)) if v else None
                for k, v in diffs.items()
            },
        }
        return out
