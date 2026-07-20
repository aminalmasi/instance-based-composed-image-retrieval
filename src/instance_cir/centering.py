from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np


def leave_one_instance_out_centers(
    features: np.ndarray,
    instance_ids: Sequence[Hashable],
    *,
    block_size: int = 16_384,
) -> dict[Hashable, np.ndarray]:
    """Compute the database mean excluding every sample of the requested instance."""
    values = np.asarray(features)
    labels = np.asarray(instance_ids)
    if values.ndim != 2 or values.shape[0] != labels.shape[0]:
        raise ValueError("features and instance IDs must align")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    unique_ids, codes = np.unique(labels, return_inverse=True)
    count = values.shape[0]
    dimension = values.shape[1]
    total = np.zeros(dimension, dtype=np.float64)
    group_sums = np.zeros((len(unique_ids), dimension), dtype=np.float64)
    group_counts = np.bincount(codes, minlength=len(unique_ids))
    for start in range(0, count, block_size):
        end = min(start + block_size, count)
        block = np.asarray(values[start:end], dtype=np.float32)
        block_codes = codes[start:end]
        total += block.sum(axis=0, dtype=np.float64)
        for code in np.unique(block_codes):
            group_sums[code] += block[block_codes == code].sum(axis=0, dtype=np.float64)

    centers: dict[Hashable, np.ndarray] = {}
    for code, instance_id in enumerate(unique_ids):
        remaining = count - int(group_counts[code])
        if remaining <= 0:
            raise ValueError("leave-one-instance-out centering needs at least two instances")
        centers[instance_id] = ((total - group_sums[code]) / remaining).astype(np.float32)
    return centers


def center_for_instance(
    features: np.ndarray,
    instance_id: Hashable,
    centers: dict[Hashable, np.ndarray],
) -> np.ndarray:
    """Subtract the matching leave-one-instance-out center."""
    if instance_id not in centers:
        raise KeyError(f"missing center for instance {instance_id!r}")
    return np.asarray(features, dtype=np.float32) - centers[instance_id]
