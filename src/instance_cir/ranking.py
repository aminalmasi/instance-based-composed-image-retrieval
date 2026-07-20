from __future__ import annotations

import numpy as np


def stable_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Return deterministic descending score indices, preserving input order for ties."""
    values = np.asarray(scores)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if k <= 0:
        raise ValueError("k must be positive")
    safe = np.nan_to_num(values.astype(np.float64, copy=False), nan=-np.inf)
    return np.argsort(-safe, kind="mergesort")[: min(k, safe.size)]


def cosine_scores(query: np.ndarray, candidates: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute cosine similarity between one query and a candidate matrix."""
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.ndim != 2 or candidates.shape[1] != query.shape[0]:
        raise ValueError("candidate and query dimensions do not match")
    query = query / max(float(np.linalg.norm(query)), eps)
    candidates = candidates / np.clip(np.linalg.norm(candidates, axis=1, keepdims=True), eps, None)
    return candidates @ query
