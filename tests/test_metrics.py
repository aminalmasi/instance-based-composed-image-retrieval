import numpy as np
import pytest

from instance_cir.metrics import MetricProfile, average_precision_at_k, evaluate_rankings


def test_ap_at_k_uses_precision_at_positive_ranks() -> None:
    relevance = [True, False, True, False]
    expected = (1.0 / 1.0 + 2.0 / 3.0) / 2.0
    assert average_precision_at_k(relevance, 2, k=4) == pytest.approx(expected)


def test_metric_profiles_expose_denominator_difference() -> None:
    relevance = np.ones(100, dtype=bool)
    thesis = average_precision_at_k(relevance, 127, k=100, profile=MetricProfile.THESIS)
    all_relevant = average_precision_at_k(relevance, 127, k=100, profile=MetricProfile.ALL_RELEVANT)
    assert thesis == pytest.approx(1.0)
    assert all_relevant == pytest.approx(100 / 127)


def test_macro_map_balances_instances() -> None:
    result = evaluate_rankings(
        [[True], [True], [False]],
        [1, 1, 1],
        ["many-query-instance", "many-query-instance", "other-instance"],
        k=1,
    )
    assert result.map_at_k == pytest.approx(2 / 3)
    assert result.macro_map_at_k == pytest.approx(0.5)
    assert result.recall_at_k == pytest.approx(2 / 3)
