from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from instance_cir.centering import leave_one_instance_out_centers
from instance_cir.data.icir import IcirDataset
from instance_cir.features import l2_normalize
from instance_cir.fusion import product_fusion
from instance_cir.metrics import MetricProfile, RetrievalMetrics, evaluate_rankings
from instance_cir.ranking import stable_top_k


@dataclass(frozen=True)
class FirstStageOutput:
    rankings: pd.DataFrame
    per_query: pd.DataFrame
    metrics: RetrievalMetrics


def run_first_stage(
    dataset: IcirDataset,
    query_image_features: np.ndarray,
    contextualized_text_features: np.ndarray,
    database_image_features: np.ndarray,
    context_corpus_features: np.ndarray,
    *,
    top_k: int = 100,
    metric_profile: MetricProfile = MetricProfile.THESIS,
    features_are_normalized: bool = True,
) -> FirstStageOutput:
    """Run centered-image times contextualized-text SigLIP-style retrieval."""
    query_image = l2_normalize(query_image_features)
    query_text = l2_normalize(contextualized_text_features)
    database_image = np.asarray(database_image_features)
    if not features_are_normalized:
        database_image = l2_normalize(database_image)
    corpus = l2_normalize(context_corpus_features)
    if query_image.shape != query_text.shape:
        raise ValueError("query image and contextualized text features must align")
    if len(query_image) != len(dataset.queries):
        raise ValueError("query features do not align with query metadata")
    if len(database_image) != len(dataset.database):
        raise ValueError("database features do not align with database metadata")
    if database_image.shape[1] != query_image.shape[1] or corpus.shape[1] != query_image.shape[1]:
        raise ValueError("all feature dimensions must match")

    instance_series = dataset.database["instance_id"]
    if isinstance(instance_series.dtype, pd.CategoricalDtype):
        instance_codes = instance_series.cat.codes.to_numpy(dtype=np.int32)
        centers_by_code = leave_one_instance_out_centers(database_image, instance_codes)
        centers = {
            str(instance): centers_by_code[code]
            for code, instance in enumerate(instance_series.cat.categories)
        }
    else:
        instance_ids: Sequence[Hashable] = instance_series.tolist()
        centers = leave_one_instance_out_centers(database_image, instance_ids)
    text_center = corpus.mean(axis=0, dtype=np.float64).astype(np.float32)

    ranking_rows: list[dict] = []
    query_rows: list[dict] = []
    relevance_rows: list[np.ndarray] = []
    relevant_counts: list[int] = []
    metric_instances: list[str] = []

    for query_index, query in dataset.queries.iterrows():
        pool = dataset.pool_for_query(int(query_index))
        instance_id = str(query["instance_id"])
        candidate_indices = pool.candidate_indices
        candidate_features = np.asarray(database_image[candidate_indices], dtype=np.float32)
        candidate_features = candidate_features - centers[instance_id]
        centered_query_image = query_image[query_index] - centers[instance_id]
        centered_query_text = query_text[query_index] - text_center
        image_scores = candidate_features @ centered_query_image
        text_scores = candidate_features @ centered_query_text
        fused_scores = product_fusion(image_scores, text_scores)
        local_order = stable_top_k(fused_scores, top_k)
        ranked_database_indices = candidate_indices[local_order]
        ranked_positive = pool.positive_mask[local_order]

        relevance_rows.append(ranked_positive)
        relevant_counts.append(len(pool.positive_ids))
        metric_instances.append(instance_id)
        query_rows.append(
            {
                "query_id": query["query_id"],
                "instance_id": instance_id,
                "modification": query["modification"],
                "num_candidates": len(candidate_indices),
                "num_relevant": len(pool.positive_ids),
            }
        )
        for rank, (database_index, local_index, is_positive) in enumerate(
            zip(ranked_database_indices, local_order, ranked_positive), start=1
        ):
            database_row = dataset.database.iloc[int(database_index)]
            ranking_rows.append(
                {
                    "query_id": query["query_id"],
                    "instance_id": instance_id,
                    "modification": query["modification"],
                    "rank": rank,
                    "num_relevant": len(pool.positive_ids),
                    "database_index": int(database_index),
                    "database_id": database_row["database_id"],
                    "image_path": database_row["image_path"],
                    "is_positive": bool(is_positive),
                    "image_score": float(image_scores[local_index]),
                    "text_score": float(text_scores[local_index]),
                    "product_score": float(fused_scores[local_index]),
                }
            )

    metrics = evaluate_rankings(
        relevance_rows,
        relevant_counts,
        metric_instances,
        k=top_k,
        profile=metric_profile,
    )
    query_df = pd.DataFrame(query_rows)
    query_df["ap_at_k"] = [item.ap_at_k for item in metrics.per_query]
    query_df["recall_at_k"] = [item.recall_at_k for item in metrics.per_query]
    return FirstStageOutput(pd.DataFrame(ranking_rows), query_df, metrics)
