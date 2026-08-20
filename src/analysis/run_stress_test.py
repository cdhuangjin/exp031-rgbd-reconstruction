"""Deterministic module-level stress tests for E3 pressure conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

for _candidate in (Path(__file__).resolve().parents[1] / "02_代码", Path("LOCAL_EXPERIMENT_ROOT/src/code")):
    if _candidate.exists():
        sys.path.insert(0, str(_candidate))
from small_modules import depth_validity_weights, pose_frame_weight


def run() -> dict[str, object]:
    depth_rates = [0.0, 0.1, 0.3, 0.5, 0.7]
    depth_rows = []
    for invalid_rate in depth_rates:
        total = 1000
        invalid = int(total * invalid_rate)
        values = [1.0] * (total - invalid) + [0.0] * invalid
        weights = depth_validity_weights(values, min_depth=0.3, max_depth=5.0)
        depth_rows.append(
            {"invalid_rate": invalid_rate, "accepted_fraction": sum(weights) / total}
        )

    motions = [0.0, 0.05, 0.2, 0.5, 1.0]
    pose_rows = [
        {
            "translation_delta": motion,
            "rotation_delta_deg": 5.0,
            "reprojection_error": 0.05,
            "weight": pose_frame_weight(motion, 5.0, 0.05),
        }
        for motion in motions
    ]
    reprojection_rows = [
        {
            "reprojection_error": error,
            "weight": pose_frame_weight(0.2, 5.0, error),
        }
        for error in [0.0, 0.05, 0.1, 0.2, 0.5]
    ]
    return {
        "protocol": "E3 module response stress test",
        "depth_validity": depth_rows,
        "pose_translation": pose_rows,
        "pose_reprojection": reprojection_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

