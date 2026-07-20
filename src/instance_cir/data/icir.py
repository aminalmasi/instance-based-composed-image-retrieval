from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

QUERY_COLUMNS = ["relative_path", "modification", "instance_id"]
DATABASE_COLUMNS = ["relative_path", "label", "instance_id"]


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


@dataclass(frozen=True)
class QueryPool:
    candidate_indices: np.ndarray
    positive_mask: np.ndarray
    positive_ids: frozenset[object]


@dataclass
class IcirDataset:
    root: Path
    queries: pd.DataFrame
    database: pd.DataFrame

    def __post_init__(self) -> None:
        required_query = {"query_id", "image_path", "modification", "instance_id"}
        required_database = {"database_id", "image_path", "label", "instance_id"}
        if not required_query.issubset(self.queries.columns):
            raise ValueError(
                f"query metadata is missing {required_query - set(self.queries.columns)}"
            )
        if not required_database.issubset(self.database.columns):
            raise ValueError(
                f"database metadata is missing {required_database - set(self.database.columns)}"
            )

        self.queries = self.queries.copy().reset_index(drop=True)
        self.database = self.database.copy().reset_index(drop=True)
        self.queries["modification_key"] = self.queries["modification"].map(normalize_text)
        self.database["label_key"] = self.database["label"].map(normalize_text).astype("category")
        self.database["instance_id"] = self.database["instance_id"].astype("category")
        self._positive_indices = {
            key: group.index.to_numpy(dtype=np.int64)
            for key, group in self.database.groupby(
                ["instance_id", "label_key"], sort=False, observed=True
            )
        }
        negative_rows = self.database[self.database["label_key"].eq("hn")]
        self._negative_indices = {
            key: group.index.to_numpy(dtype=np.int64)
            for key, group in negative_rows.groupby("instance_id", sort=False, observed=True)
        }

    def pool_for_query(self, query_index: int) -> QueryPool:
        query = self.queries.iloc[int(query_index)]
        key = (query["instance_id"], query["modification_key"])
        positives = np.asarray(self._positive_indices.get(key, []), dtype=np.int64)
        negatives = np.asarray(self._negative_indices.get(query["instance_id"], []), dtype=np.int64)
        if positives.size == 0:
            raise ValueError(f"query {query['query_id']} has no relevant database candidates")
        candidate_indices = np.unique(np.concatenate([positives, negatives]))
        positive_mask = np.isin(candidate_indices, positives, assume_unique=True)
        positive_ids = frozenset(self.database.iloc[positives]["database_id"].tolist())
        return QueryPool(candidate_indices, positive_mask, positive_ids)


def _resolve_paths(root: Path, values: pd.Series) -> pd.Series:
    """Resolve metadata paths lexically without one filesystem syscall per image."""
    paths = values.astype(str).copy()
    relative = ~paths.str.startswith("/")
    prefix = str(root.absolute()).rstrip("/") + "/"
    paths.loc[relative] = prefix + paths.loc[relative].str.lstrip("/")
    return paths


def load_icir_folder(root: str | Path, *, validate_images: bool = False) -> IcirDataset:
    """Load the official three-column local-folder ICIR metadata."""
    root = Path(root).expanduser().resolve()
    query_csv = root / "query_files.csv"
    database_csv = root / "database_files.csv"
    if not query_csv.is_file() or not database_csv.is_file():
        raise FileNotFoundError(f"expected query_files.csv and database_files.csv under {root}")

    queries = pd.read_csv(query_csv, header=None, names=QUERY_COLUMNS, dtype=str).fillna("")
    database = pd.read_csv(database_csv, header=None, names=DATABASE_COLUMNS, dtype=str).fillna("")
    queries.insert(0, "query_id", [f"q_{index:05d}" for index in range(len(queries))])
    database.insert(0, "database_id", np.arange(len(database), dtype=np.int64))
    queries["image_path"] = _resolve_paths(root.parent, queries["relative_path"])
    database["image_path"] = _resolve_paths(root.parent, database["relative_path"])
    queries.drop(columns="relative_path", inplace=True)
    database.drop(columns="relative_path", inplace=True)

    if validate_images:
        missing_query = queries.loc[~queries["image_path"].map(Path.is_file), "image_path"]
        missing_database = database.loc[~database["image_path"].map(Path.is_file), "image_path"]
        if not missing_query.empty or not missing_database.empty:
            raise FileNotFoundError(
                f"missing {len(missing_query)} query and {len(missing_database)} database images"
            )
    return IcirDataset(root=root, queries=queries, database=database)
