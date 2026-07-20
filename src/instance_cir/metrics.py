from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np


class MetricProfile(str, Enum):
    """Supported AP conventions.

    THESIS reproduces the code path used for the final reported thesis results.
    ALL_RELEVANT follows the denominator written in the thesis equation.
    """

    THESIS = "thesis"
    ALL_RELEVANT = "all_relevant"


@dataclass(frozen=True)
class QueryMetrics:
    ap_at_k: float
    recall_at_k: float
    hits_at_k: int
    total_relevant: int


@dataclass(frozen=True)
class RetrievalMetrics:
    map_at_k: float
    macro_map_at_k: float
    recall_at_k: float
    num_queries: int
    num_instances: int
    per_query: tuple[QueryMetrics, ...]


def average_precision_at_k(
    relevance: Sequence[bool] | np.ndarray,
    total_relevant: int,
    *,
    k: int = 100,
    profile: MetricProfile = MetricProfile.THESIS,
) -> float:
    """Compute AP@k from a ranked binary relevance sequence."""
    if k <= 0:
        raise ValueError("k must be positive")
    if total_relevant < 0:
        raise ValueError("total_relevant cannot be negative")
    if total_relevant == 0:
        return 0.0

    flags = np.asarray(relevance, dtype=bool).reshape(-1)[:k]
    hit_ranks = np.flatnonzero(flags) + 1
    if hit_ranks.size == 0:
        return 0.0

    precision_at_hits = np.arange(1, hit_ranks.size + 1, dtype=np.float64) / hit_ranks
    if profile == MetricProfile.THESIS:
        denominator = min(total_relevant, k)
    elif profile == MetricProfile.ALL_RELEVANT:
        denominator = total_relevant
    else:  # pragma: no cover - protected by the enum in normal use
        raise ValueError(f"unsupported metric profile: {profile}")
    return float(precision_at_hits.sum() / denominator)


def recall_at_k(
    relevance: Sequence[bool] | np.ndarray,
    total_relevant: int,
    *,
    k: int = 100,
) -> float:
    """Compute Recall@k using every relevant candidate as the denominator."""
    if k <= 0:
        raise ValueError("k must be positive")
    if total_relevant < 0:
        raise ValueError("total_relevant cannot be negative")
    if total_relevant == 0:
        return 0.0
    flags = np.asarray(relevance, dtype=bool).reshape(-1)[:k]
    return float(flags.sum() / total_relevant)


def evaluate_rankings(
    ranked_relevance: Sequence[Sequence[bool] | np.ndarray],
    total_relevant: Sequence[int],
    instance_ids: Sequence[Hashable],
    *,
    k: int = 100,
    profile: MetricProfile = MetricProfile.THESIS,
) -> RetrievalMetrics:
    """Evaluate query-level, global, and instance-balanced retrieval metrics."""
    if not (len(ranked_relevance) == len(total_relevant) == len(instance_ids)):
        raise ValueError("rankings, relevant counts, and instance IDs must have equal length")
    if not ranked_relevance:
        raise ValueError("at least one query is required")

    query_metrics: list[QueryMetrics] = []
    grouped_aps: dict[Hashable, list[float]] = {}
    for relevance, n_relevant, instance_id in zip(ranked_relevance, total_relevant, instance_ids):
        flags = np.asarray(relevance, dtype=bool).reshape(-1)[:k]
        ap = average_precision_at_k(flags, int(n_relevant), k=k, profile=profile)
        recall = recall_at_k(flags, int(n_relevant), k=k)
        query_metrics.append(
            QueryMetrics(
                ap_at_k=ap,
                recall_at_k=recall,
                hits_at_k=int(flags.sum()),
                total_relevant=int(n_relevant),
            )
        )
        grouped_aps.setdefault(instance_id, []).append(ap)

    aps = np.asarray([item.ap_at_k for item in query_metrics], dtype=np.float64)
    recalls = np.asarray([item.recall_at_k for item in query_metrics], dtype=np.float64)
    instance_means = np.asarray(
        [np.mean(values) for values in grouped_aps.values()], dtype=np.float64
    )
    return RetrievalMetrics(
        map_at_k=float(aps.mean()),
        macro_map_at_k=float(instance_means.mean()),
        recall_at_k=float(recalls.mean()),
        num_queries=len(query_metrics),
        num_instances=len(grouped_aps),
        per_query=tuple(query_metrics),
    )


def relevant_flags(ranked_ids: Iterable[str], positive_ids: set[str]) -> np.ndarray:
    """Convert ranked candidate identifiers to binary relevance flags."""
    return np.fromiter((candidate in positive_ids for candidate in ranked_ids), dtype=bool)
