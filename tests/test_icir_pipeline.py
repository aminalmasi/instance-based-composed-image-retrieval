from pathlib import Path

import numpy as np
import pandas as pd

from instance_cir.data.icir import IcirDataset
from instance_cir.pipeline.stage1 import run_first_stage


def test_tiny_first_stage_uses_query_local_pool() -> None:
    queries = pd.DataFrame(
        [
            {
                "query_id": "q0",
                "image_path": "query.jpg",
                "modification": "red",
                "instance_id": "chair",
            }
        ]
    )
    database = pd.DataFrame(
        [
            {
                "database_id": "positive",
                "image_path": "p.jpg",
                "label": "red",
                "instance_id": "chair",
            },
            {
                "database_id": "hard-negative",
                "image_path": "n.jpg",
                "label": "hn",
                "instance_id": "chair",
            },
            {"database_id": "other", "image_path": "o.jpg", "label": "red", "instance_id": "car"},
        ]
    )
    dataset = IcirDataset(Path("."), queries, database)
    query_image = np.asarray([[1.0, 0.0]], dtype=np.float32)
    query_text = np.asarray([[1.0, 0.0]], dtype=np.float32)
    database_image = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    context = np.asarray([[0.0, 1.0]], dtype=np.float32)

    result = run_first_stage(dataset, query_image, query_text, database_image, context, top_k=2)
    assert result.rankings["database_id"].tolist() == ["positive", "hard-negative"]
    assert result.metrics.map_at_k == 1.0
    assert result.metrics.recall_at_k == 1.0
