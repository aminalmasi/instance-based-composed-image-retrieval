#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ("mAP@100", "Macro-mAP@100", "Recall@100")


def check_result(name: str, actual_path: Path, target: dict, tolerance: float) -> list[str]:
    actual = json.loads(actual_path.read_text(encoding="utf-8"))
    failures = []
    for metric in METRICS:
        if metric not in actual:
            failures.append(f"{name}: output is missing {metric}")
            continue
        difference = abs(float(actual[metric]) - float(target[metric]))
        if difference > tolerance:
            failures.append(
                f"{name}: {metric}={actual[metric]:.4f}, expected={target[metric]:.4f}, "
                f"difference={difference:.4f}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, default=Path("reproducibility/thesis_targets.json"))
    parser.add_argument("--first-stage", type=Path, required=True)
    parser.add_argument("--route-a", type=Path, required=True)
    parser.add_argument("--route-b", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()
    targets = json.loads(args.targets.read_text(encoding="utf-8"))["targets"]
    failures = []
    failures.extend(
        check_result("first_stage", args.first_stage, targets["first_stage"], args.tolerance)
    )
    failures.extend(check_result("route_a", args.route_a, targets["route_a"], args.tolerance))
    failures.extend(check_result("route_b", args.route_b, targets["route_b"], args.tolerance))
    if failures:
        print("Regression check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("All thesis metrics match their recorded targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
