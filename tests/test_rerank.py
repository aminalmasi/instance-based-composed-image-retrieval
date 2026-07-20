import numpy as np
import pandas as pd

from instance_cir.pipeline.rerank import rerank_shortlist


def test_reranker_reorders_fixed_candidates_without_changing_recall() -> None:
    rankings = pd.DataFrame(
        [
            {
                "query_id": "q",
                "instance_id": "i",
                "rank": 1,
                "num_relevant": 1,
                "is_positive": False,
                "image_score": 0.9,
                "text_score": 0.9,
            },
            {
                "query_id": "q",
                "instance_id": "i",
                "rank": 2,
                "num_relevant": 1,
                "is_positive": True,
                "image_score": 0.8,
                "text_score": 0.8,
            },
        ]
    )
    extra = {
        "route_image": np.asarray([0.1, 0.9]),
        "route_text": np.asarray([0.1, 0.9]),
    }
    output = rerank_shortlist(
        rankings,
        extra,
        variables=["siglip2_image", "siglip2_text", "route_image", "route_text"],
        top_k=2,
    )
    assert output.rankings.iloc[0]["is_positive"]
    assert output.metrics.map_at_k == 1.0
    assert output.metrics.recall_at_k == 1.0
