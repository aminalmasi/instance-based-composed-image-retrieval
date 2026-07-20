from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from instance_cir.data.icir import load_icir_folder
from instance_cir.features import FeatureManifest
from instance_cir.metrics import MetricProfile
from instance_cir.pipeline.rerank import rerank_shortlist
from instance_cir.pipeline.stage1 import run_first_stage


def _metric_summary(metrics, top_k: int) -> dict:
    return {
        f"mAP@{top_k}": 100.0 * metrics.map_at_k,
        f"Macro-mAP@{top_k}": 100.0 * metrics.macro_map_at_k,
        f"Recall@{top_k}": 100.0 * metrics.recall_at_k,
        "num_queries": metrics.num_queries,
        "num_instances": metrics.num_instances,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def command_validate_data(args: argparse.Namespace) -> int:
    dataset = load_icir_folder(args.data_root, validate_images=args.check_images)
    pool_sizes = []
    positive_counts = []
    for query_index in range(len(dataset.queries)):
        pool = dataset.pool_for_query(query_index)
        pool_sizes.append(len(pool.candidate_indices))
        positive_counts.append(len(pool.positive_ids))
    summary = {
        "data_root": str(dataset.root),
        "num_queries": len(dataset.queries),
        "num_database_images": len(dataset.database),
        "num_instances": int(dataset.queries["instance_id"].nunique()),
        "minimum_pool_size": int(min(pool_sizes)),
        "maximum_pool_size": int(max(pool_sizes)),
        "minimum_positive_count": int(min(positive_counts)),
        "maximum_positive_count": int(max(positive_counts)),
    }
    print(json.dumps(summary, indent=2))
    return 0


def command_stage1(args: argparse.Namespace) -> int:
    dataset = load_icir_folder(args.data_root)
    manifest = FeatureManifest.load(args.feature_manifest)
    query_image, query_text, database_image = manifest.open_arrays()
    context_features = np.load(args.context_features, mmap_mode="r")
    result = run_first_stage(
        dataset,
        query_image,
        query_text,
        database_image,
        context_features,
        top_k=args.top_k,
        metric_profile=MetricProfile(args.metric_profile),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.rankings.to_csv(args.output_dir / "topk_rankings.csv", index=False)
    result.per_query.to_csv(args.output_dir / "per_query_metrics.csv", index=False)
    summary = _metric_summary(result.metrics, args.top_k)
    summary.update(
        {
            "method": "siglip2_centered_image_x_contextualized_text",
            "model_id": manifest.model_id,
            "model_revision": manifest.model_revision,
            "metric_profile": args.metric_profile,
            "candidate_protocol": "instance_specific_positives_plus_hard_negatives",
        }
    )
    _write_json(args.output_dir / "summary_metrics.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def command_rerank(args: argparse.Namespace) -> int:
    import pandas as pd

    rankings = pd.read_csv(args.rankings)
    score_bank = np.load(args.score_bank)
    extra_scores = {name: score_bank[name] for name in score_bank.files}
    variables = [value.strip() for value in args.variables.split(",") if value.strip()]
    result = rerank_shortlist(
        rankings,
        extra_scores,
        variables=variables,
        top_k=args.top_k,
        metric_profile=MetricProfile(args.metric_profile),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.rankings.to_csv(args.output_dir / "reranked_topk.csv", index=False)
    summary = _metric_summary(result.metrics, args.top_k)
    summary.update(
        {
            "variables": variables,
            "metric_profile": args.metric_profile,
            "candidate_protocol": "fixed_first_stage_shortlist",
        }
    )
    _write_json(args.output_dir / "summary_metrics.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def command_train_probe(args: argparse.Namespace) -> int:
    from instance_cir.training.trainer import train_probe

    summary = train_probe(args.config)
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="instance-cir")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data", help="validate ICIR metadata and pools")
    validate.add_argument("--data-root", type=Path, required=True)
    validate.add_argument("--check-images", action="store_true")
    validate.set_defaults(handler=command_validate_data)

    stage1 = subparsers.add_parser("stage1", help="run the final SigLIP2 first stage")
    stage1.add_argument("--data-root", type=Path, required=True)
    stage1.add_argument("--feature-manifest", type=Path, required=True)
    stage1.add_argument("--context-features", type=Path, required=True)
    stage1.add_argument("--output-dir", type=Path, required=True)
    stage1.add_argument("--top-k", type=int, default=100)
    stage1.add_argument(
        "--metric-profile", choices=[item.value for item in MetricProfile], default="thesis"
    )
    stage1.set_defaults(handler=command_stage1)

    rerank = subparsers.add_parser("rerank", help="rerank an aligned fixed shortlist")
    rerank.add_argument("--rankings", type=Path, required=True)
    rerank.add_argument("--score-bank", type=Path, required=True)
    rerank.add_argument("--variables", required=True)
    rerank.add_argument("--output-dir", type=Path, required=True)
    rerank.add_argument("--top-k", type=int, default=100)
    rerank.add_argument(
        "--metric-profile", choices=[item.value for item in MetricProfile], default="thesis"
    )
    rerank.set_defaults(handler=command_rerank)

    train = subparsers.add_parser("train-probe", help="train an ILIAS linear projection head")
    train.add_argument("--config", type=Path, required=True)
    train.set_defaults(handler=command_train_probe)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
