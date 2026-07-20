import numpy as np

from instance_cir.fusion import product_fusion
from instance_cir.ranking import stable_top_k


def test_product_clips_negative_evidence() -> None:
    actual = product_fusion(np.asarray([0.8, -0.2]), np.asarray([0.5, 0.9]))
    np.testing.assert_allclose(actual, [0.4, 0.0])


def test_stable_ranking_preserves_order_for_ties() -> None:
    order = stable_top_k(np.asarray([0.5, 0.8, 0.8, 0.2]), 4)
    np.testing.assert_array_equal(order, [1, 2, 0, 3])
