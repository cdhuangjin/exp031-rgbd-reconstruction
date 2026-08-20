from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(payload: dict) -> dict:
    rows = payload.get("results", [])
    levels = sorted({r["noise_level"] for r in rows}, key=lambda x: ["clean", "mild", "moderate", "severe"].index(x))
    scenes = sorted({r["sequence"] for r in rows})
    methods = ["baseline", "m1", "m2", "m3", "full"]
    expected = len(levels) * len(scenes) * len(methods)
    finite = all(np.isfinite(float(r["heldout_depth_rmse"])) for r in rows)
    by_key = {(r["noise_level"], r["sequence"], r["variant"]): r for r in rows}
    summary = {}
    for level in levels:
        level_summary = {}
        for scene in scenes:
            base = by_key[(level, scene, "baseline")]["heldout_depth_rmse"]
            scene_summary = {}
            for method in methods:
                value = by_key[(level, scene, method)]["heldout_depth_rmse"]
                scene_summary[method] = {
                    "rmse": float(value),
                    "delta_vs_baseline": float(value - base),
                    "coverage": float(by_key[(level, scene, method)]["heldout_depth_coverage"]),
                }
            level_summary[scene] = scene_summary
        summary[level] = level_summary
    aggregate = {}
    rng = np.random.default_rng(31031)
    for level in levels:
        for method in methods:
            values = [by_key[(level, scene, method)]["heldout_depth_rmse"] for scene in scenes]
            deltas = [summary[level][scene][method]["delta_vs_baseline"] for scene in scenes]
            samples = rng.choice(np.asarray(deltas, dtype=float), size=(10000, len(deltas)), replace=True)
            ci_low, ci_high = np.quantile(samples.mean(axis=1), [0.025, 0.975])
            aggregate.setdefault(level, {})[method] = {
                "mean_rmse": float(np.mean(values)),
                "std_rmse": float(np.std(values)),
                "mean_delta_vs_baseline": float(np.mean(deltas)),
                "improved_scenes": int(sum(d < 0 for d in deltas)),
                "bootstrap95_ci_delta": [float(ci_low), float(ci_high)],
            }
    return {
        "status": "PASS" if len(rows) == expected and finite and len(by_key) == expected else "FAIL",
        "rows": len(rows),
        "expected_rows": expected,
        "levels": levels,
        "scenes": scenes,
        "finite_metrics": finite,
        "aggregate": aggregate,
        "by_scene": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
