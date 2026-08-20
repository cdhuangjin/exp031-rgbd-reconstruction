from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

from active_selection import CandidateView, select_next_view
from calibration import calibration_radius, should_stop


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def generate_scene(seed: int, candidate_count: int) -> list[CandidateView]:
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least 2")
    rng = random.Random(seed)
    return [
        CandidateView(
            view_id=f"v{index:03d}",
            nominal_gain=0.25 + rng.random() * 0.75,
            pose_gain_std=0.01 + rng.random() * 0.30,
            uncovered_risk=rng.random(),
            cost=0.05 + rng.random() * 0.95,
        )
        for index in range(candidate_count)
    ]


def run(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    rng = random.Random(seed)
    candidates = generate_scene(seed, int(config["candidate_count"]))
    residuals = [0.03, 0.07, 0.05, 0.09, 0.04]
    radius = calibration_radius(residuals, alpha=float(config["alpha"]))
    quality = float(config["initial_quality"])
    records = []
    remaining = list(candidates)
    start = time.perf_counter()

    for step in range(int(config["max_steps"])):
        selected = select_next_view(
            remaining,
            mode=str(config["mode"]),
            lambda_pose=float(config["lambda_pose"]),
            lambda_cost=float(config["lambda_cost"]),
            lambda_cov=float(config["lambda_cov"]),
        )
        remaining = [candidate for candidate in remaining if candidate.view_id != selected.view_id]
        quality = min(1.0, quality + 0.08 * selected.nominal_gain + rng.uniform(-0.01, 0.01))
        stop = should_stop(
            predicted_quality=quality,
            radius=radius,
            target_quality=float(config["target_quality"]),
            uncovered_risk=selected.uncovered_risk,
            max_risk=float(config["max_risk"]),
        )
        records.append(
            {
                "step": step + 1,
                "selected_view": selected.view_id,
                "nominal_gain": selected.nominal_gain,
                "pose_gain_std": selected.pose_gain_std,
                "uncovered_risk": selected.uncovered_risk,
                "cost": selected.cost,
                "quality": quality,
                "quality_lower_bound": quality - radius,
                "stop": stop,
            }
        )
        if stop or not remaining:
            break

    return {
        "config_hash": _config_hash(config),
        "seed": seed,
        "mode": config["mode"],
        "calibration_radius": radius,
        "steps": records,
        "final_quality": records[-1]["quality"],
        "stopped": records[-1]["stop"],
        "wall_time_seconds": time.perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=12)
    args = parser.parse_args()
    config = {
        "seed": args.seed,
        "candidate_count": args.candidate_count,
        "max_steps": args.max_steps,
        "mode": "full_risk",
        "alpha": 0.1,
        "lambda_pose": 1.0,
        "lambda_cost": 0.1,
        "lambda_cov": 0.5,
        "initial_quality": 0.35,
        "target_quality": 0.70,
        "max_risk": 0.20,
    }
    result = run(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"config": config, "result": result}, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
