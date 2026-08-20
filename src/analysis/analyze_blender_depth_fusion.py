from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(payload: dict) -> dict:
    rows = payload["results"]
    scenes = sorted({r["sequence"] for r in rows})
    methods = ["baseline", "m1", "m2", "m3", "full"]
    by_key = {(r["sequence"], r["variant"]): r for r in rows}
    scene_summary = {}
    for scene in scenes:
        base = by_key[(scene, "baseline")]["heldout_depth_rmse"]
        scene_summary[scene] = {}
        for method in methods:
            row = by_key[(scene, method)]
            value = float(row["heldout_depth_rmse"])
            scene_summary[scene][method] = {
                "rmse": value,
                "delta_vs_baseline": float(value - base) if np.isfinite(value) and np.isfinite(base) else None,
                "coverage": float(row["heldout_depth_coverage"]) if np.isfinite(row["heldout_depth_coverage"]) else None,
                "finite": bool(np.isfinite(value)),
            }
    aggregate = {}
    for method in methods:
        values = [scene_summary[s][method]["rmse"] for s in scenes if scene_summary[s][method]["finite"]]
        deltas = [scene_summary[s][method]["delta_vs_baseline"] for s in scenes if scene_summary[s][method]["delta_vs_baseline"] is not None]
        aggregate[method] = {"finite_scenes": len(values), "mean_rmse": float(np.mean(values)) if values else None, "mean_delta_vs_baseline": float(np.mean(deltas)) if deltas else None}
    return {"status": "PASS_WITH_DOMAIN_FAILURES" if len(rows) == len(scenes) * len(methods) else "FAIL", "rows": len(rows), "expected_rows": len(scenes) * len(methods), "scenes": scenes, "aggregate": aggregate, "by_scene": scene_summary, "domain_failure_scenes": [s for s in scenes if not scene_summary[s]["full"]["finite"]]}


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
