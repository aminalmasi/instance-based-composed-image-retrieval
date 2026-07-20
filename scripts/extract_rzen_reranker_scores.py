#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from instance_cir.features import l2_normalize
from instance_cir.models.rzen import RzenEncoder

DEFAULT_INSTRUCTION = (
    "Represent the candidate image for exact instance-level retrieval. "
    "Use the provided instance descriptor as an attention guide for the literal identity. "
    "Prioritize discriminative identity cues and ignore background, style, lighting, pose, "
    "and state changes unless they help identify the instance. Instance descriptor: "
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--query-image-features", type=Path, required=True)
    parser.add_argument("--query-text-features", type=Path, required=True)
    parser.add_argument("--query-metadata", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--start-batch-size", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=24)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--max-image-tokens", type=int, default=768)
    args = parser.parse_args()

    rankings = pd.read_csv(args.rankings).fillna("")
    descriptors = pd.read_csv(args.descriptors).fillna("")
    query_meta = pd.read_csv(args.query_metadata).fillna("").reset_index(drop=True)
    descriptor_column = (
        "line2_instance_descriptor"
        if "line2_instance_descriptor" in descriptors.columns
        else "instance_descriptor"
    )
    descriptor_lookup = descriptors.set_index("query_id")[descriptor_column].astype(str).to_dict()
    query_lookup = {str(query_id): index for index, query_id in enumerate(query_meta["query_id"])}
    query_image = l2_normalize(np.load(args.query_image_features, mmap_mode="r"))
    query_text = l2_normalize(np.load(args.query_text_features, mmap_mode="r"))
    encoder = RzenEncoder(
        args.model_root,
        dtype=args.dtype,
        max_image_tokens=args.max_image_tokens,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_scores = np.empty(len(rankings), dtype=np.float32)
    text_scores = np.empty(len(rankings), dtype=np.float32)

    for query_id, group in rankings.groupby("query_id", sort=False):
        query_id = str(query_id)
        if query_id not in descriptor_lookup or not descriptor_lookup[query_id].strip():
            raise KeyError(f"missing instance descriptor for {query_id}")
        if query_id not in query_lookup:
            raise KeyError(f"missing RZEn query features for {query_id}")
        row_indices = group.index.to_numpy(np.int64)
        descriptor = descriptor_lookup[query_id]
        candidate_features = encoder.encode_fused_adaptive(
            group["image_path"].astype(str).tolist(),
            [descriptor] * len(group),
            args.instruction,
            start_batch_size=args.start_batch_size,
            max_batch_size=args.max_batch_size,
        )
        query_index = query_lookup[query_id]
        image_scores[row_indices] = candidate_features @ query_image[query_index]
        text_scores[row_indices] = candidate_features @ query_text[query_index]

    np.savez_compressed(
        args.output_dir / "rzen_scores.npz",
        rzen_image=image_scores,
        rzen_text=text_scores,
    )
    manifest = {
        "rankings": str(args.rankings.resolve()),
        "descriptors": str(args.descriptors.resolve()),
        "instruction": args.instruction,
        "candidate_rows": len(rankings),
        "query_count": int(rankings["query_id"].nunique()),
        "variables": ["rzen_image", "rzen_text"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
