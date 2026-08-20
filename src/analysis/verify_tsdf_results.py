"""Fail-closed audit for the three-window sparse-TSDF reference results."""

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


def audit(root: Path) -> dict[str, object]:
    files = sorted(root.glob("tsdf_baseline_s*_cuda.json"))
    errors: list[str] = []
    checks = {"three_window_files": len(files) == 3, "all_cuda": True, "all_metrics_finite": True}
    if len(files) != 3:
        errors.append(f"expected 3 files, found {len(files)}")
    seen: set[tuple[str, str, str]] = set()
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        rows = payload.get("results", [])
        if len(rows) != 5:
            errors.append(f"{file.name}: expected 5 rows, found {len(rows)}")
        for row in rows:
            key = (file.name, str(row.get("sequence")), str(row.get("variant")))
            if key in seen:
                errors.append(f"duplicate row: {key}")
            seen.add(key)
            if row.get("sequence") not in SCENES:
                errors.append(f"unexpected scene: {row.get('sequence')}")
            if row.get("variant") != "sparse_tsdf":
                errors.append(f"unexpected variant: {row.get('variant')}")
            if row.get("device") != "cuda":
                checks["all_cuda"] = False
                errors.append(f"non-CUDA row: {key}")
            for metric in ("heldout_depth_rmse", "heldout_depth_mae", "heldout_depth_coverage"):
                if not math.isfinite(float(row.get(metric, float("nan")))):
                    checks["all_metrics_finite"] = False
                    errors.append(f"non-finite {metric}: {key}")
    checks["complete_scene_rows"] = len(seen) == 15
    if not checks["complete_scene_rows"]:
        errors.append(f"expected 15 unique scene/window rows, found {len(seen)}")
    return {
        "files": [file.name for file in files],
        "row_count": len(seen),
        "checks": checks,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }


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
