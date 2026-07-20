import json
from pathlib import Path


def test_final_thesis_targets_are_versioned() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "reproducibility/thesis_targets.json").read_text())
    assert payload["query_count"] == 1883
    assert payload["targets"]["first_stage"]["mAP@100"] == 59.64
    assert payload["targets"]["route_a"]["mAP@100"] == 66.05
    assert payload["targets"]["route_b"]["mAP@100"] == 66.42
