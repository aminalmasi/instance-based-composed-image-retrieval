from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "group_id",
    "instance_id",
    "split",
    "role",
    "embedding_bank",
    "embedding_row",
}


@dataclass(frozen=True)
class EmbeddingRef:
    bank: str
    row: int


@dataclass(frozen=True)
class TrainingGroup:
    group_id: str
    instance_id: str
    split: str
    query: EmbeddingRef
    positives: tuple[EmbeddingRef, ...]
    negatives: dict[str, tuple[EmbeddingRef, ...]]


class EmbeddingBanks:
    """Memory-mapped embedding arrays addressed by symbolic names."""

    def __init__(self, paths: Mapping[str, str | Path]) -> None:
        if not paths:
            raise ValueError("at least one embedding bank is required")
        self.arrays = {
            name: np.load(Path(path).expanduser().resolve(), mmap_mode="r")
            for name, path in paths.items()
        }
        dimensions = {array.shape[1] for array in self.arrays.values() if array.ndim == 2}
        if len(dimensions) != 1 or any(array.ndim != 2 for array in self.arrays.values()):
            raise ValueError("embedding banks must be 2D arrays with one shared dimension")
        self.embedding_dim = dimensions.pop()

    def get(self, ref: EmbeddingRef) -> np.ndarray:
        if ref.bank not in self.arrays:
            raise KeyError(f"unknown embedding bank: {ref.bank}")
        array = self.arrays[ref.bank]
        if ref.row < 0 or ref.row >= len(array):
            raise IndexError(f"row {ref.row} is outside embedding bank {ref.bank}")
        return np.asarray(array[ref.row], dtype=np.float32)


def load_training_groups(path: str | Path, *, top_positives: int = 3) -> list[TrainingGroup]:
    table = pd.read_csv(path).fillna("")
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"supervision table is missing columns: {sorted(missing)}")
    if top_positives <= 0:
        raise ValueError("top_positives must be positive")

    groups: list[TrainingGroup] = []
    for group_id, frame in table.groupby("group_id", sort=False):
        query_rows = frame[frame["role"].eq("query")]
        if len(query_rows) != 1:
            raise ValueError(f"group {group_id} must contain exactly one query")
        positive_rows = frame[frame["role"].eq("positive")].copy()
        negative_rows = frame[frame["role"].eq("negative")].copy()
        if positive_rows.empty:
            raise ValueError(f"group {group_id} has no positives")
        if negative_rows.empty:
            raise ValueError(f"group {group_id} has no negatives")
        if "positive_rank" in positive_rows.columns:
            positive_rows["positive_rank"] = pd.to_numeric(
                positive_rows["positive_rank"], errors="coerce"
            ).fillna(np.inf)
            positive_rows = positive_rows.sort_values("positive_rank", kind="mergesort")
        positive_rows = positive_rows.head(top_positives)

        query = query_rows.iloc[0]
        negative_groups: dict[str, tuple[EmbeddingRef, ...]] = {}
        if "negative_hardness" not in negative_rows.columns:
            negative_rows["negative_hardness"] = "hard"
        for hardness, rows in negative_rows.groupby("negative_hardness", sort=False):
            label = str(hardness) or "hard"
            negative_groups[label] = tuple(
                EmbeddingRef(str(row.embedding_bank), int(row.embedding_row))
                for row in rows.itertuples(index=False)
            )

        groups.append(
            TrainingGroup(
                group_id=str(group_id),
                instance_id=str(query["instance_id"]),
                split=str(query["split"]),
                query=EmbeddingRef(str(query["embedding_bank"]), int(query["embedding_row"])),
                positives=tuple(
                    EmbeddingRef(str(row.embedding_bank), int(row.embedding_row))
                    for row in positive_rows.itertuples(index=False)
                ),
                negatives=negative_groups,
            )
        )
    return groups


class GroupSampler:
    """Deterministic epoch-varying anchor, positive, and hard-negative sampling."""

    hardness_order = ("super_hard", "very_hard", "hard")

    def __init__(
        self,
        groups: list[TrainingGroup],
        banks: EmbeddingBanks,
        *,
        split: str,
        k_negatives: int,
        seed: int,
        negative_sampling: str = "hardest_first",
    ) -> None:
        self.groups = [group for group in groups if group.split == split]
        self.banks = banks
        self.k_negatives = int(k_negatives)
        self.seed = int(seed)
        self.negative_sampling = negative_sampling
        self.epoch = 0
        if not self.groups:
            raise ValueError(f"no groups found for split {split!r}")
        if self.k_negatives <= 0:
            raise ValueError("k_negatives must be positive")

    def __len__(self) -> int:
        return len(self.groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def sample(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        group = self.groups[index]
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 97)
        positive = group.positives[int(rng.integers(len(group.positives)))]
        ordered_labels = [label for label in self.hardness_order if label in group.negatives]
        ordered_labels.extend(label for label in group.negatives if label not in ordered_labels)
        all_negatives = [ref for label in ordered_labels for ref in group.negatives[label]]
        if self.negative_sampling == "uniform":
            selected = rng.choice(
                len(all_negatives),
                size=self.k_negatives,
                replace=len(all_negatives) < self.k_negatives,
            )
            negative_refs = [all_negatives[int(i)] for i in selected]
        elif self.negative_sampling == "hardest_first":
            shuffled: list[EmbeddingRef] = []
            for label in ordered_labels:
                refs = list(group.negatives[label])
                rng.shuffle(refs)
                shuffled.extend(refs)
            if len(shuffled) < self.k_negatives:
                selected = rng.choice(
                    len(shuffled), size=self.k_negatives - len(shuffled), replace=True
                )
                shuffled.extend(shuffled[int(i)] for i in selected)
            negative_refs = shuffled[: self.k_negatives]
        else:
            raise ValueError(f"unsupported negative sampling mode: {self.negative_sampling}")
        return (
            self.banks.get(group.query),
            self.banks.get(positive),
            np.stack([self.banks.get(ref) for ref in negative_refs]),
        )
