from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def contextualized_prompts(
    modification: str,
    subjects: Sequence[str],
    *,
    modification_first: bool = False,
) -> list[str]:
    """Compose a short modification with generic subjects as in BASIC contextualization."""
    modification = " ".join(str(modification).split())
    if not modification:
        raise ValueError("modification cannot be empty")
    clean_subjects = [" ".join(str(subject).split()) for subject in subjects]
    clean_subjects = [subject for subject in clean_subjects if subject]
    if not clean_subjects:
        raise ValueError("at least one contextualization subject is required")
    if modification_first:
        return [f"{modification} {subject}" for subject in clean_subjects]
    return [f"{subject} {modification}" for subject in clean_subjects]


def mean_context_embedding(
    modification: str,
    subjects: Sequence[str],
    encode_texts: Callable[[Sequence[str]], np.ndarray],
    *,
    modification_first: bool = False,
    normalize: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Encode contextualized prompts and return their mean representation."""
    prompts = contextualized_prompts(modification, subjects, modification_first=modification_first)
    embeddings = np.asarray(encode_texts(prompts), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(prompts):
        raise ValueError("text encoder returned an unexpected shape")
    mean = embeddings.mean(axis=0)
    if normalize:
        mean = mean / max(float(np.linalg.norm(mean)), eps)
    return mean.astype(np.float32)
