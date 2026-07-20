#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from instance_cir.contextualization import contextualized_prompts
from instance_cir.data.icir import load_icir_folder
from instance_cir.features import sha256_file
from instance_cir.models.siglip import SiglipEncoder


def read_subjects(path: Path, limit: int) -> list[str]:
    frame = pd.read_csv(path, header=None).fillna("")
    subjects = [str(value).strip() for value in frame.iloc[:, 0] if str(value).strip()]
    if subjects and subjects[0].lower() in {"text", "object_name", "subject"}:
        subjects = subjects[1:]
    return subjects[:limit] if limit > 0 else subjects


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-id", default="google/siglip2-so400m-patch16-384")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--context-count", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()

    dataset = load_icir_folder(args.data_root, validate_images=True)
    subjects = read_subjects(args.subjects, args.context_count)
    encoder = SiglipEncoder(
        args.model_id, revision=args.revision, device=args.device, dtype=args.dtype
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_image = encoder.encode_images(
        dataset.queries["image_path"].tolist(), batch_size=args.image_batch_size
    )
    database_image = encoder.encode_images(
        dataset.database["image_path"].tolist(), batch_size=args.image_batch_size
    )
    subject_features = encoder.encode_texts(subjects, batch_size=args.text_batch_size)

    contextualized = []
    for modification in dataset.queries["modification"].tolist():
        prompts = contextualized_prompts(modification, subjects)
        prompt_features = encoder.encode_texts(prompts, batch_size=args.text_batch_size)
        mean = prompt_features.astype(np.float32).mean(axis=0)
        mean /= max(float(np.linalg.norm(mean)), 1e-12)
        contextualized.append(mean)
    query_text = np.stack(contextualized).astype(np.float16)

    query_image_path = args.output_dir / "query_image_features.npy"
    query_text_path = args.output_dir / "query_context_text_features.npy"
    database_image_path = args.output_dir / "database_image_features.npy"
    context_path = args.output_dir / "context_subject_features.npy"
    np.save(query_image_path, query_image)
    np.save(query_text_path, query_text)
    np.save(database_image_path, database_image)
    np.save(context_path, subject_features)
    manifest = {
        "model_id": args.model_id,
        "model_revision": args.revision,
        "embedding_dim": int(query_image.shape[1]),
        "dtype": "float16",
        "query_image_features": query_image_path.name,
        "query_text_features": query_text_path.name,
        "database_image_features": database_image_path.name,
        "context_subject_features": context_path.name,
        "metadata": {
            "context_count": len(subjects),
            "query_count": len(query_image),
            "database_count": len(database_image),
            "query_metadata_sha256": sha256_file(dataset.root / "query_files.csv"),
            "database_metadata_sha256": sha256_file(dataset.root / "database_files.csv"),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
