"""Fail-closed audit for the three-window held-out CUDA result set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SCENES = {
    "freiburg1_desk",
    "freiburg1_desk2",
    "freiburg1_room",
    "freiburg2_xyz",
    "freiburg3_long_office_household",
}
TARGET_SCENES = {"freiburg1_room", "freiburg2_xyz", "freiburg3_long_office_household"}
METHODS = {"baseline", "m1", "m2", "m3", "full"}


def audit(root: Path) -> dict[str, object]:
    files = sorted(root.glob("heldout_cuda_s*.json"))
    checks: dict[str, bool] = {}
    errors: list[str] = []
    checks["three_window_files"] = len(files) == 3
    if not checks["three_window_files"]:
        errors.append(f"expected 3 heldout_cuda_s*.json files, found {len(files)}")
    rows_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    thresholds = []
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        rows = payload.get("results", [])
        thresholds.append(payload.get("config", {}).get("threshold"))
        if len(rows) != 25:
            errors.append(f"{file.name}: expected 25 rows, found {len(rows)}")
        for row in rows:
            key = (file.name, str(row.get("sequence")), str(row.get("variant")))
            if key in rows_by_key:
                errors.append(f"duplicate row: {key}")
            rows_by_key[key] = row
            if row.get("sequence") not in SCENES:
                errors.append(f"unexpected scene: {row.get('sequence')}")
            if row.get("variant") not in METHODS:
                errors.append(f"unexpected method: {row.get('variant')}")
            if row.get("device") != "cuda":
                errors.append(f"non-CUDA row: {key}")
            for metric in ("heldout_depth_rmse", "heldout_depth_mae", "heldout_depth_coverage"):
                if not math.isfinite(float(row.get(metric, float("nan")))):
                    errors.append(f"non-finite {metric}: {key}")

    complete_windows = all(
        all((file.name, scene, method) in rows_by_key for scene in SCENES for method in METHODS)
        for file in files
    )
    checks["complete_25_rows_per_window"] = complete_windows
    if not complete_windows:
        errors.append("at least one window is missing a scene/method row")
    checks["all_metrics_finite"] = not any("non-finite" in e for e in errors)
    checks["all_cuda"] = not any("non-CUDA" in e for e in errors)

    deltas: dict[str, list[float]] = {scene: [] for scene in sorted(TARGET_SCENES)}
    for file in files:
        for scene in sorted(TARGET_SCENES):
            baseline = rows_by_key[(file.name, scene, "baseline")]
            full = rows_by_key[(file.name, scene, "full")]
            deltas[scene].append(float(full["heldout_depth_rmse"]) - float(baseline["heldout_depth_rmse"]))
    mean_delta = {scene: sum(values) / len(values) for scene, values in deltas.items() if values}
    improved_scene_count = sum(value < 0 for value in mean_delta.values())
    checks["full_improves_at_least_two_scene_means"] = improved_scene_count >= 2
    if not checks["full_improves_at_least_two_scene_means"]:
        errors.append(f"Full improves only {improved_scene_count} scene means")
    checks["thresholds_present"] = all(isinstance(value, (int, float)) and value > 0 for value in thresholds)
    if not checks["thresholds_present"]:
        errors.append("missing or invalid source threshold")
    result = {
        "files": [file.name for file in files],
        "row_count": len(rows_by_key),
        "thresholds": thresholds,
        "full_minus_baseline_rmse_mean": mean_delta,
        "improved_scene_count": improved_scene_count,
        "checks": checks,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
