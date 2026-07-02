"""Embedding-based rewards: API-free replacements for correctness + graph_utility.

Paper §4.1.3 specifies:
  correctness    = judge(gold, candidate_answer)              -> 0.30 weight
  graph_utility  = judge(gold, judge_reconstruct(graph_json)) -> 0.25 weight

We replace both with cosine similarity in a shared embedding space (BGE-base-en-v1.5,
same model paper uses for backtracking analysis in §4.3). This keeps the reward
in the same semantic frame the paper uses for evaluation.

Embedding reward design:
  correctness   = cosine(emb(candidate_answer), emb(gold_answer))
  graph_utility = cosine(emb(graph_json_as_text), emb(gold_answer))

Both are deterministic, zero-API, fast (batched).
"""
from __future__ import annotations

import json
import re
import threading
from typing import Any, List, Optional

import numpy as np
import torch

_GRAPH_JSON_RE = re.compile(r"<graph_json>\s*(\{.*?\})\s*</graph_json>", flags=re.DOTALL)
_THINK_END = "</think>"


def extract_post_think_answer(full_output: str) -> str:
    if _THINK_END not in full_output:
        return full_output.strip()
    idx = full_output.rfind(_THINK_END)
    return full_output[idx + len(_THINK_END):].strip()


def extract_graph_json_str(full_output: str) -> Optional[str]:
    m = _GRAPH_JSON_RE.search(full_output)
    return m.group(1) if m else None


def render_graph_as_text(graph_json_str: str) -> str:
    """Render a graph_json payload as natural-language lines for embedding.

    Example: "SilkFiber constrains Tension" — concatenation is what sentence-BERT
    style embedders handle well. Paper §4.1.3 reward (diversity) uses the same
    "source relation target" rendering.
    """
    try:
        obj = json.loads(graph_json_str)
    except Exception:
        return ""
    lines: List[str] = []
    nodes = obj.get("nodes", []) or []
    edges = obj.get("edges", []) or []
    for n in nodes:
        nid = n.get("id") or ""
        ntype = n.get("type") or ""
        if nid:
            lines.append(f"{nid} : {ntype}".strip(" :"))
    for e in edges:
        s = e.get("source") or ""
        r = e.get("relation") or ""
        t = e.get("target") or ""
        if s and r and t:
            lines.append(f"{s} {r} {t}")
    return "\n".join(lines)


class Embedder:
    """Lazy-loaded sentence-transformer with batched encoding."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(
                        self.model_name, device=self.device
                    )
                    self._model.eval()
        return self._model

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        model = self._ensure_model()
        emb = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32, copy=False)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


_DEFAULT_EMBEDDER: Optional[Embedder] = None


def get_default_embedder() -> Embedder:
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        _DEFAULT_EMBEDDER = Embedder()
    return _DEFAULT_EMBEDDER


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine on already-normalized vectors; falls back to general cosine if not."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def score_correctness_embedding(
    embedder: Embedder,
    candidate_answer: str,
    gold_answer: str,
) -> float:
    """cosine(candidate, gold) in [0,1] rescaled.

    Embedding cosines for semantically related text typically land in [0.4, 0.9].
    We rescale linearly so that 0.4 -> 0.0 and 0.9 -> 1.0, which gives the reward
    useful dynamic range. (The exact affine transform is a knob; this is documented
    in judge/rubric.md and matches paper's typical judge score distribution.)
    """
    if not candidate_answer or not gold_answer:
        return 0.0
    a = embedder.encode_one(candidate_answer)
    b = embedder.encode_one(gold_answer)
    raw = cosine(a, b)
    return max(0.0, min(1.0, (raw - 0.4) / 0.5))


def score_graph_utility_embedding(
    embedder: Embedder,
    graph_json_str: Optional[str],
    gold_answer: str,
) -> float:
    """cosine(graph_rendered, gold). Information-bottleneck proxy: if the graph
    contains the same semantic content as the gold answer, cosine is high."""
    if not graph_json_str or not gold_answer:
        return 0.0
    graph_text = render_graph_as_text(graph_json_str)
    if not graph_text.strip():
        return 0.0
    a = embedder.encode_one(graph_text)
    b = embedder.encode_one(gold_answer)
    raw = cosine(a, b)
    # Graphs are sparser than full prose; use a slightly lower floor.
    return max(0.0, min(1.0, (raw - 0.30) / 0.55))


def batch_correctness(
    embedder: Embedder,
    pairs: List[tuple[str, str]],
) -> List[float]:
    """Batched cosine for many (candidate, gold) pairs."""
    cands = [p[0] for p in pairs]
    golds = [p[1] for p in pairs]
    if not cands:
        return []
    ce = embedder.encode(cands)
    ge = embedder.encode(golds)
    raw = (ce * ge).sum(axis=1)
    scaled = np.clip((raw - 0.4) / 0.5, 0.0, 1.0)
    return scaled.tolist()


def batch_graph_utility(
    embedder: Embedder,
    pairs: List[tuple[Optional[str], str]],
) -> List[float]:
    """Batched cosine for (graph_json_str, gold) pairs."""
    graphs = []
    golds = []
    keep_idx = []
    for i, (gj, ga) in enumerate(pairs):
        if not gj or not ga:
            continue
        text = render_graph_as_text(gj)
        if not text.strip():
            continue
        graphs.append(text)
        golds.append(ga)
        keep_idx.append(i)
    if not graphs:
        return [0.0] * len(pairs)
    ge = embedder.encode(graphs)
    gde = embedder.encode(golds)
    raw = (ge * gde).sum(axis=1)
    scaled = np.clip((raw - 0.30) / 0.55, 0.0, 1.0)
    out = [0.0] * len(pairs)
    for k, idx in enumerate(keep_idx):
        out[idx] = float(scaled[k])
    return out
