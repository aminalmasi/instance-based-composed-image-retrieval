#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def hardness_from_bucket(bucket_k: int) -> str:
    if bucket_k < 50:
        return "super_hard"
    if bucket_k < 150:
        return "very_hard"
    return "hard"


def build_supervision(
    split_csv: Path,
    positivity_csv: Path,
    negativity_csv: Path,
    distractor_metadata_csv: Path,
) -> pd.DataFrame:
    split = pd.read_csv(split_csv).fillna("")
    positives = pd.read_csv(positivity_csv).fillna("")
    negatives = pd.read_csv(negativity_csv).fillna("")
    distractor_metadata = pd.read_csv(distractor_metadata_csv).fillna("")
    if "distractor_global_row" not in distractor_metadata.columns:
        raise ValueError("distractor metadata needs a distractor_global_row column")
    distractor_row = {
        int(global_row): int(array_row)
        for array_row, global_row in enumerate(distractor_metadata["distractor_global_row"])
    }
    split_lookup = split.set_index("query_row")
    positive_groups = {int(key): frame for key, frame in positives.groupby("query_row")}
    negative_groups = {int(key): frame for key, frame in negatives.groupby("query_row")}
    rows: list[dict] = []

    for query_row, split_row in split_lookup.iterrows():
        query_row = int(query_row)
        group_id = str(split_row.get("query_record_key", f"query_{query_row}"))
        common = {
            "group_id": group_id,
            "instance_id": str(split_row["instance_id"]),
            "split": str(split_row["split"]),
        }
        rows.append(
            {
                **common,
                "sample_id": group_id,
                "role": "query",
                "embedding_bank": "queries",
                "embedding_row": query_row,
                "positive_rank": "",
                "negative_hardness": "",
            }
        )
        for positive in positive_groups.get(query_row, pd.DataFrame()).itertuples(index=False):
            rows.append(
                {
                    **common,
                    "sample_id": str(positive.positive_record_key),
                    "role": "positive",
                    "embedding_bank": "positives",
                    "embedding_row": int(positive.positive_core_index),
                    "positive_rank": int(positive.siglip2_rank),
                    "negative_hardness": "",
                }
            )
        negative_group = negative_groups.get(query_row, pd.DataFrame())
        for negative in negative_group.itertuples(index=False):
            if str(negative.negative_source) != "distractor":
                continue
            global_row = int(float(negative.negative_distractor_global_row))
            if global_row not in distractor_row:
                continue
            rows.append(
                {
                    **common,
                    "sample_id": f"distractor_{global_row}",
                    "role": "negative",
                    "embedding_bank": "distractors",
                    "embedding_row": distractor_row[global_row],
                    "positive_rank": "",
                    "negative_hardness": hardness_from_bucket(int(negative.bucket_k)),
                }
            )
    result = pd.DataFrame(rows)
    valid_groups = result.groupby("group_id")["role"].agg(
        lambda values: {"query", "positive", "negative"}.issubset(set(values))
    )
    return result[result["group_id"].isin(valid_groups[valid_groups].index)].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--positivity-csv", type=Path, required=True)
    parser.add_argument("--negativity-csv", type=Path, required=True)
    parser.add_argument("--distractor-metadata", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    table = build_supervision(
        args.split_csv,
        args.positivity_csv,
        args.negativity_csv,
        args.distractor_metadata,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    print(table.groupby(["split", "role"], sort=False).size().rename("samples").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
