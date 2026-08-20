"""Summarize wall-clock fields from cloud result JSON files."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path


def summarize(root: Path, pattern: str) -> dict[str, float | int | str]:
    files = sorted(root.glob(pattern))
    values = [float(row["wall_time_seconds"]) for file in files for row in json.loads(file.read_text())["results"]]
    return {
        "pattern": pattern,
        "files": len(files),
        "rows": len(values),
        "mean_wall_time_seconds": round(statistics.mean(values), 6) if values else None,
        "max_wall_time_seconds": round(max(values), 6) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "summaries": [
            summarize(args.root, "heldout_cuda_s*.json"),
            summarize(args.root, "extended_cuda_s*_m48.json"),
            summarize(args.root, "open3d_tsdf_s*.json"),
            summarize(args.root, "tsdf_baseline_s*_cuda.json"),
        ]
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
