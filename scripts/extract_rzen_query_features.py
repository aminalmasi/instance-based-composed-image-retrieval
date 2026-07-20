#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from instance_cir.data.icir import load_icir_folder
from instance_cir.features import l2_normalize
from instance_cir.models.rzen import RzenEncoder

IMAGE_INSTRUCTION = "Represent the given image for retrieval."
TEXT_INSTRUCTION = "Represent the modification text for retrieving matching images. "


def adaptive_single_modality(
    encoder: RzenEncoder,
    values: list[str],
    *,
    modality: str,
    instruction: str,
    start_batch_size: int,
    max_batch_size: int,
) -> np.ndarray:
    torch = encoder.torch
    chunks = []
    cursor = 0
    batch_size = min(start_batch_size, max_batch_size, len(values))
    while cursor < len(values):
        current = min(batch_size, len(values) - cursor)
        try:
            kwargs = {
                "instruction": instruction,
                "batch_size": current,
                "show_progress_bar": False,
            }
            if modality == "image":
                kwargs["images"] = values[cursor : cursor + current]
            else:
                kwargs["texts"] = values[cursor : cursor + current]
            with torch.inference_mode():
                embeddings = encoder.model.get_fused_embeddings(**kwargs)
            chunks.append(embeddings.detach().float().cpu().numpy())
            cursor += current
        except RuntimeError as error:
            message = str(error).lower()
            if "out of memory" not in message and "cuda" not in message:
                raise
            torch.cuda.empty_cache()
            if current <= 1:
                raise
            batch_size = max(1, current // 2)
    return l2_normalize(np.concatenate(chunks)).astype(np.float16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-batch-size", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=24)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()
    dataset = load_icir_folder(args.data_root, validate_images=True)
    encoder = RzenEncoder(args.model_root, dtype=args.dtype)
    image_features = adaptive_single_modality(
        encoder,
        dataset.queries["image_path"].astype(str).tolist(),
        modality="image",
        instruction=IMAGE_INSTRUCTION,
        start_batch_size=args.start_batch_size,
        max_batch_size=args.max_batch_size,
    )
    text_features = adaptive_single_modality(
        encoder,
        dataset.queries["modification"].astype(str).tolist(),
        modality="text",
        instruction=TEXT_INSTRUCTION,
        start_batch_size=args.start_batch_size,
        max_batch_size=args.max_batch_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "query_image_features.npy", image_features)
    np.save(args.output_dir / "query_text_features.npy", text_features)
    dataset.queries.to_csv(args.output_dir / "query_metadata.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_root": str(args.model_root.resolve()),
                "image_instruction": IMAGE_INSTRUCTION,
                "text_instruction": TEXT_INSTRUCTION,
                "query_count": len(dataset.queries),
                "embedding_dim": int(image_features.shape[1]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
