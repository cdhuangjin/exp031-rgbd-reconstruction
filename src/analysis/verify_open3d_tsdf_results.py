"""Fail-closed audit for the three-window Open3D TSDF reference results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SCENES = {"freiburg1_desk", "freiburg1_desk2", "freiburg1_room", "freiburg2_xyz", "freiburg3_long_office_household"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.root.glob("open3d_tsdf_s*.json"))
    errors: list[str] = []
    if len(files) != 3:
        errors.append(f"expected 3 files, found {len(files)}")
    seen: set[tuple[str, str]] = set()
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        if payload.get("config", {}).get("open3d_version") != "0.19.0":
            errors.append(f"unexpected Open3D version: {file.name}")
        rows = payload.get("results", [])
        if len(rows) != 5:
            errors.append(f"{file.name}: expected 5 rows, found {len(rows)}")
        for row in rows:
            key = (file.name, str(row.get("sequence")))
            if key in seen:
                errors.append(f"duplicate row: {key}")
            seen.add(key)
            if row.get("sequence") not in SCENES:
                errors.append(f"unexpected scene: {row.get('sequence')}")
            if row.get("variant") != "open3d_tsdf" or row.get("device") != "cpu_open3d":
                errors.append(f"unexpected implementation fields: {key}")
            for metric in ("heldout_depth_rmse", "heldout_depth_mae", "heldout_depth_coverage"):
                if not math.isfinite(float(row.get(metric, float("nan")))):
                    errors.append(f"non-finite {metric}: {key}")
    result = {
        "files": [file.name for file in files],
        "row_count": len(seen),
        "errors": errors,
        "status": "PASS" if not errors and len(seen) == 15 else "FAIL",
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
