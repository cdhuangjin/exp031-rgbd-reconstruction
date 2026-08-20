"""Audit and summarize the expanded max-frames=48 CUDA experiment."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


SCENES = {
    "freiburg1_desk",
    "freiburg1_desk2",
    "freiburg1_room",
    "freiburg2_xyz",
    "freiburg3_long_office_household",
}
TARGETS = {"freiburg1_room", "freiburg2_xyz", "freiburg3_long_office_household"}
METHODS = {"baseline", "m1", "m2", "m3", "full"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.root.glob("extended_cuda_s*_m48.json"))
    errors: list[str] = []
    rows: dict[tuple[str, str, str], dict] = {}
    if len(files) != 3:
        errors.append(f"expected 3 files, found {len(files)}")
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        batch = payload.get("results", [])
        if len(batch) != 25:
            errors.append(f"{file.name}: expected 25 rows, found {len(batch)}")
        for row in batch:
            key = (file.name, str(row.get("sequence")), str(row.get("variant")))
            if key in rows:
                errors.append(f"duplicate row: {key}")
            rows[key] = row
            if row.get("sequence") not in SCENES or row.get("variant") not in METHODS:
                errors.append(f"unexpected identity: {key}")
            if row.get("device") != "cuda":
                errors.append(f"non-CUDA row: {key}")
            for metric in ("heldout_depth_rmse", "heldout_depth_mae", "heldout_depth_coverage"):
                if not math.isfinite(float(row.get(metric, float("nan")))):
                    errors.append(f"non-finite {metric}: {key}")
            if int(row.get("heldout_frames", 0)) < 5:
                errors.append(f"too few heldout frames: {key}")
    complete = all((file.name, scene, method) in rows for file in files for scene in SCENES for method in METHODS)
    if not complete:
        errors.append("incomplete scene/method/window matrix")

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for scene in sorted(SCENES):
        aggregate[scene] = {}
        for method in sorted(METHODS):
            values = [rows[(file.name, scene, method)] for file in files]
            aggregate[scene][method] = {
                "rmse_mean": statistics.mean(float(v["heldout_depth_rmse"]) for v in values),
                "rmse_popstd": statistics.pstdev(float(v["heldout_depth_rmse"]) for v in values),
                "mae_mean": statistics.mean(float(v["heldout_depth_mae"]) for v in values),
                "coverage_mean": statistics.mean(float(v["heldout_depth_coverage"]) for v in values),
                "heldout_frames_mean": statistics.mean(int(v["heldout_frames"]) for v in values),
            }
    comparisons = {}
    for scene in sorted(TARGETS):
        baseline = aggregate[scene]["baseline"]
        full = aggregate[scene]["full"]
        comparisons[scene] = {
            "full_minus_baseline_rmse": full["rmse_mean"] - baseline["rmse_mean"],
            "full_minus_baseline_coverage": full["coverage_mean"] - baseline["coverage_mean"],
            "baseline_rmse": baseline["rmse_mean"],
            "full_rmse": full["rmse_mean"],
            "baseline_coverage": baseline["coverage_mean"],
            "full_coverage": full["coverage_mean"],
            "coverage_adjusted_score_penalty_1m_baseline": baseline["rmse_mean"] + (1.0 - baseline["coverage_mean"]),
            "coverage_adjusted_score_penalty_1m_full": full["rmse_mean"] + (1.0 - full["coverage_mean"]),
            "coverage_adjusted_delta_penalty_1m": (
                full["rmse_mean"] + (1.0 - full["coverage_mean"])
                - baseline["rmse_mean"] - (1.0 - baseline["coverage_mean"])
            ),
            "improved_windows": sum(
                float(rows[(file.name, scene, "full")]["heldout_depth_rmse"])
                < float(rows[(file.name, scene, "baseline")]["heldout_depth_rmse"])
                for file in files
            ),
            "window_count": len(files),
        }
    result = {
        "files": [f.name for f in files],
        "row_count": len(rows),
        "checks": {
            "three_window_files": len(files) == 3,
            "complete_25_rows_per_window": complete,
            "all_metrics_finite": not any("non-finite" in e for e in errors),
            "all_cuda": not any("non-CUDA" in e for e in errors),
            "heldout_frames_at_least_five": not any("too few" in e for e in errors),
        },
        "aggregate": aggregate,
        "target_comparisons": comparisons,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
