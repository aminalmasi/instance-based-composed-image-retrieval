from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def nonnegative(score: np.ndarray) -> np.ndarray:
    """Clip cosine-derived evidence before multiplicative fusion."""
    return np.clip(np.asarray(score, dtype=np.float32), 0.0, None)


def product_fusion(*scores: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Multiply aligned nonnegative score vectors."""
    if len(scores) < 2:
        raise ValueError("product fusion requires at least two score vectors")
    arrays = [nonnegative(score) + eps for score in scores]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays[1:]):
        raise ValueError("all score vectors must have the same shape")
    return np.multiply.reduce(arrays)


def named_product(
    available_scores: Mapping[str, np.ndarray],
    variables: Sequence[str],
    *,
    eps: float = 0.0,
) -> np.ndarray:
    """Fuse a configured subset of named score vectors."""
    missing = [name for name in variables if name not in available_scores]
    if missing:
        raise KeyError(f"missing score variables: {', '.join(missing)}")
    return product_fusion(*(available_scores[name] for name in variables), eps=eps)
