from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from instance_cir.fusion import named_product
from instance_cir.metrics import MetricProfile, RetrievalMetrics, evaluate_rankings
from instance_cir.ranking import stable_top_k


@dataclass(frozen=True)
class RerankOutput:
    rankings: pd.DataFrame
    metrics: RetrievalMetrics


def rerank_shortlist(
    first_stage_rankings: pd.DataFrame,
    extra_scores: Mapping[str, np.ndarray],
    *,
    variables: Sequence[str],
    top_k: int = 100,
    metric_profile: MetricProfile = MetricProfile.THESIS,
) -> RerankOutput:
    """Rerank a fixed first-stage shortlist by a configured product of aligned factors."""
    required = {
        "query_id",
        "instance_id",
        "rank",
        "is_positive",
        "image_score",
        "text_score",
    }
    if not required.issubset(first_stage_rankings.columns):
        raise ValueError(f"first-stage rankings are missing {required - set(first_stage_rankings)}")
    row_count = len(first_stage_rankings)
    if any(np.asarray(values).reshape(-1).size != row_count for values in extra_scores.values()):
        raise ValueError("every extra score vector must align with the ranking rows")

    frame = first_stage_rankings.copy().reset_index(drop=True)
    available: dict[str, np.ndarray] = {
        "siglip2_image": frame["image_score"].to_numpy(np.float32),
        "siglip2_text": frame["text_score"].to_numpy(np.float32),
    }
    available.update(
        {
            name: np.asarray(values, dtype=np.float32).reshape(-1)
            for name, values in extra_scores.items()
        }
    )
    frame["rerank_score"] = named_product(available, variables)

    ranked_groups: list[pd.DataFrame] = []
    relevance: list[np.ndarray] = []
    relevant_counts: list[int] = []
    instance_ids: list[str] = []
    for _query_id, group in frame.groupby("query_id", sort=False):
        order = stable_top_k(group["rerank_score"].to_numpy(), top_k)
        ranked = group.iloc[order].copy().reset_index(drop=True)
        ranked["old_rank"] = ranked["rank"].astype(int)
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        ranked_groups.append(ranked)
        flags = ranked["is_positive"].to_numpy(bool)
        relevance.append(flags)
        # A fixed top-100 reranker only sees relevant candidates recalled by stage one.
        # The full denominator is carried in num_relevant when available.
        if "num_relevant" in group.columns:
            total_relevant = int(group["num_relevant"].iloc[0])
        else:
            total_relevant = int(group["is_positive"].sum())
        relevant_counts.append(total_relevant)
        instance_ids.append(str(group["instance_id"].iloc[0]))

    metrics = evaluate_rankings(
        relevance,
        relevant_counts,
        instance_ids,
        k=top_k,
        profile=metric_profile,
    )
    return RerankOutput(pd.concat(ranked_groups, ignore_index=True), metrics)
