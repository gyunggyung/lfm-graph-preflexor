"""Programmatic reward components ported from upstream src/run_grpo_graph.py.

These do not call any API and are deterministic. The implementations follow the
paper's §4.1.3 formulas exactly. Source: graph-preflexor-grpo/src/run_grpo_graph.py
lines 254-692.
"""
from __future__ import annotations

import json
import re
import threading
from typing import Any, List, Optional, Tuple

# Sentinel tags (must match the dataset)
THINK_START, THINK_END = "<think>", "</think>"
BRAINSTORM_START, BRAINSTORM_END = "<brainstorm>", "</brainstorm>"
GRAPH_START, GRAPH_END = "<graph>", "</graph>"
GRAPH_JSON_START, GRAPH_JSON_END = "<graph_json>", "</graph_json>"
PATTERNS_START, PATTERNS_END = "<patterns>", "</patterns>"
SYNTHESIS_START, SYNTHESIS_END = "<synthesis>", "</synthesis>"

_GRAPH_JSON_RE = re.compile(
    r"<graph_json>\s*(\{.*?\})\s*</graph_json>", flags=re.DOTALL
)


def _find_span(haystack: str, start: str, end: str) -> Optional[Tuple[int, int]]:
    i = haystack.find(start)
    if i < 0:
        return None
    j = haystack.find(end, i + len(start))
    if j < 0:
        return None
    return (i, j + len(end))


def extract_graph_json_str(full_output: str) -> Optional[str]:
    m = _GRAPH_JSON_RE.search(full_output)
    return m.group(1) if m else None


def extract_post_think(full_output: str) -> str:
    if THINK_END not in full_output:
        return full_output.strip()
    idx = full_output.rfind(THINK_END)
    return full_output[idx + len(THINK_END):].strip()


def score_format(full_output: str) -> float:
    """Paper §4.1.3 format score. Max 1.0."""
    score = 0.0
    if THINK_START in full_output and THINK_END in full_output:
        score += 0.15
    if BRAINSTORM_START in full_output and BRAINSTORM_END in full_output:
        score += 0.10
    if GRAPH_START in full_output and GRAPH_END in full_output:
        score += 0.15

    gj_str = extract_graph_json_str(full_output)
    if gj_str is None:
        return max(0.0, min(1.0, score))

    score += 0.20

    if PATTERNS_START in full_output and PATTERNS_END in full_output:
        score += 0.15
    if SYNTHESIS_START in full_output and SYNTHESIS_END in full_output:
        score += 0.15

    try:
        obj = json.loads(gj_str)
        if isinstance(obj, dict) and isinstance(obj.get("nodes"), list) and len(obj["nodes"]) > 0:
            score += 0.10
    except Exception:
        pass

    return max(0.0, min(1.0, score))


def _parse_graph(full_output: str) -> Optional[dict]:
    gj = extract_graph_json_str(full_output)
    if gj is None:
        return None
    try:
        obj = json.loads(gj)
        if not isinstance(obj, dict):
            return None
        obj.setdefault("nodes", [])
        obj.setdefault("edges", [])
        return obj
    except Exception:
        return None


def score_graph_networkx(full_output: str) -> float:
    """Paper Eq. (5). Max 1.0."""
    try:
        import networkx as nx
    except ImportError:
        return 0.0
    obj = _parse_graph(full_output)
    if obj is None:
        return 0.0
    nodes = obj.get("nodes") or []
    edges = obj.get("edges") or []
    n = len(nodes)
    m = len(edges)
    if n == 0:
        return 0.0

    score = 0.0
    if n > 0:
        score += 0.3
    node_ids = {(n_.get("id") if isinstance(n_, dict) else n_) for n_ in nodes}

    invalid = 0
    self_loops = 0
    for e in edges:
        if not isinstance(e, dict):
            invalid += 1
            continue
        s = e.get("source")
        t = e.get("target")
        if s not in node_ids or t not in node_ids:
            invalid += 1
        elif s == t:
            self_loops += 1
    valid_edges = m - invalid
    if valid_edges > 0:
        score += 0.3
    if m > 0:
        score += 0.2 * (1 - invalid / m)
    if self_loops == 0:
        score += 0.1

    # weak connectivity
    try:
        g = nx.DiGraph()
        for nid in node_ids:
            if nid:
                g.add_node(nid)
        for e in edges:
            if isinstance(e, dict):
                g.add_edge(e.get("source"), e.get("target"))
        if nx.is_weakly_connected(g):
            score += 0.1
    except Exception:
        pass

    return max(0.0, min(1.0, score))


_DIV_LOCK = threading.Lock()
_DIV_CACHE: dict = {}


def score_graph_diversity(full_output: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    """Paper Eq. (6). Max ~1.0 (soft-capped by design)."""
    obj = _parse_graph(full_output)
    if obj is None:
        return 0.0
    items: List[str] = []
    for n_ in obj.get("nodes") or []:
        if isinstance(n_, dict) and n_.get("id"):
            items.append(str(n_["id"]))
    for e in obj.get("edges") or []:
        if isinstance(e, dict):
            s, r, t = e.get("source"), e.get("relation"), e.get("target")
            if s and r and t:
                items.append(f"{s} {r} {t}")
    m_prime = len(items)
    if m_prime < 2:
        return 0.0

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return 0.0

    with _DIV_LOCK:
        st = _DIV_CACHE.get(model_name)
        if st is None:
            st = SentenceTransformer(model_name)
            _DIV_CACHE[model_name] = st

    import numpy as np
    emb = st.encode(items, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    sim = emb @ emb.T
    mask = ~np.eye(m_prime, dtype=bool)
    s_bar = float(sim[mask].mean())

    d = max(0.0, min(1.0, 1 - (s_bar - 0.15) / 0.35))
    bonus = min(0.1, m_prime / 100.0)
    return max(0.0, min(1.0, 0.9 * d + bonus))


def score_graph_structure(full_output: str) -> float:
    """Paper §4.1.3 structure reward. Max ~1.0 (soft-capped)."""
    try:
        import networkx as nx
    except ImportError:
        return 0.0
    obj = _parse_graph(full_output)
    if obj is None:
        return 0.0
    try:
        g = nx.DiGraph()
        for n_ in obj.get("nodes") or []:
            if isinstance(n_, dict) and n_.get("id"):
                g.add_node(n_["id"])
        for e in obj.get("edges") or []:
            if isinstance(e, dict) and e.get("source") and e.get("target"):
                g.add_edge(e["source"], e["target"])
    except Exception:
        return 0.0
    n = g.number_of_nodes()
    m = g.number_of_edges()
    if n == 0:
        return 0.0

    # size: peak at 5-20 nodes
    if n < 4:
        s_size = 0.05 * n / 4
    elif n <= 20:
        s_size = 0.20
    else:
        s_size = max(0.0, 0.20 - 0.01 * (n - 20))

    # density
    rho = m / max(1, n * (n - 1))
    s_dens = min(0.20, 2 * rho)

    # internal nodes (in-deg>0 and out-deg>0)
    n_int = sum(
        1 for nd in g.nodes()
        if g.in_degree(nd) > 0 and g.out_degree(nd) > 0
    )
    s_int = 0.30 * (n_int / n)

    # depth (DAG longest path, capped at 6)
    try:
        if nx.is_directed_acyclic_graph(g):
            L = nx.dag_longest_path_length(g)
        else:
            L = 0
    except Exception:
        L = 0
    s_depth = min(0.20, 0.20 * min(L, 6) / 6)

    # weak connectivity
    s_conn = 0.10 if nx.is_weakly_connected(g) else 0.0

    return max(0.0, min(1.0, s_size + s_dens + s_int + s_depth + s_conn))
