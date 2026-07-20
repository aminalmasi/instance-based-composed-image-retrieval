from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatureManifest:
    model_id: str
    model_revision: str
    embedding_dim: int
    dtype: str
    query_image_features: Path
    query_text_features: Path
    database_image_features: Path
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> FeatureManifest:
        manifest_path = Path(path).expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = manifest_path.parent

        def resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()

        return cls(
            model_id=payload["model_id"],
            model_revision=payload.get("model_revision", "main"),
            embedding_dim=int(payload["embedding_dim"]),
            dtype=payload["dtype"],
            query_image_features=resolve(payload["query_image_features"]),
            query_text_features=resolve(payload["query_text_features"]),
            database_image_features=resolve(payload["database_image_features"]),
            metadata=payload.get("metadata", {}),
        )

    def open_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.load(self.query_image_features, mmap_mode="r"),
            np.load(self.query_text_features, mmap_mode="r"),
            np.load(self.database_image_features, mmap_mode="r"),
        )


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 1:
        return values / max(float(np.linalg.norm(values)), eps)
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), eps, None)
