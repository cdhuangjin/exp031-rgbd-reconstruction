from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from small_modules import (
    combine_frame_reliability,
    depth_validity_weights,
    pose_frame_weight,
    scene_adaptive_threshold,
)


def run_smoke(*, output: Path | None = None, seed: int = 0) -> dict[str, Any]:
    """Run a dependency-free deterministic check for the three small modules."""
    source_errors = [0.05, 0.07, 0.12, 0.09]
    threshold = scene_adaptive_threshold(source_errors, quantile=0.75)
    depth_weights = depth_validity_weights(
        [0.0, 0.4, 1.2, 4.0], min_depth=0.2, max_depth=3.0
    )
    pose_stats = [
        (0.02, 0.5, 0.01),
        (0.20, 4.0, 0.08),
        (0.05, 1.0, 0.02),
    ]
    pose_weights = [pose_frame_weight(*stats) for stats in pose_stats]
    frame_reliability = [
        combine_frame_reliability(depth_weight, pose_weight)
        for depth_weight, pose_weight in zip(depth_weights[1:], pose_weights)
    ]
    result = {
        "seed": int(seed),
        "source_threshold": threshold,
        "depth_weights": depth_weights,
        "pose_weights": pose_weights,
        "frame_reliability": frame_reliability,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v3 small-module smoke test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_smoke(output=args.output, seed=args.seed)
    print(args.output)


if __name__ == "__main__":
    main()
