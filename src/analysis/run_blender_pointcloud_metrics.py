"""Independent point-cloud metrics against held-out NeRF Synthetic depth.

Build points come only from the three build quarters. Held-out depth is used
only to form the reference point cloud for evaluation. Metrics use a 5 cm
nearest-neighbour tolerance and capped deterministic samples for auditability.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from run_blender_depth_fusion import load_scene
from small_modules import pose_frame_weight, scene_adaptive_threshold


def _cloud(item: dict, variant: str, threshold: float) -> np.ndarray:
    build, weights = [], []
    for i, record in enumerate(item["records"]):
        if i % 4 == 0:
            continue
        if variant in {"m3", "full"} and record["invalid_ratio"] > threshold:
            continue
        points = record["points"]
        if variant in {"m1", "m3", "full"}:
            points = points[(points[:, 2] > -5.0) & (points[:, 2] < 5.0)]
        weight = pose_frame_weight(record["translation_delta"], record["rotation_delta"], record["invalid_ratio"]) if variant in {"m2", "full"} else 1.0
        build.append(points)
        weights.append(np.full(len(points), weight, dtype=np.float64))
    if not build:
        return np.empty((0, 3), dtype=np.float64)
    points = np.concatenate(build)
    weights = np.concatenate(weights)
    keys = np.floor(points / 0.03).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys, points, weights = keys[order], points[order], weights[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1))]
    ends = np.r_[starts[1:], len(points)]
    sums = np.add.reduceat(points * weights[:, None], starts, axis=0)
    totals = np.add.reduceat(weights, starts)
    return sums / np.maximum(totals[:, None], 1e-12)


def _reference(item: dict, max_points: int, seed: int) -> np.ndarray:
    points = np.concatenate([r["points"] for i, r in enumerate(item["records"]) if i % 4 == 0], axis=0)
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=max_points, replace=False)]


def _metrics(cloud: np.ndarray, reference: np.ndarray, tolerance: float) -> dict:
    if len(cloud) == 0 or len(reference) == 0:
        return {"chamfer_m": float("nan"), "precision_at_5cm": 0.0, "recall_at_5cm": 0.0, "cloud_points": int(len(cloud)), "reference_points": int(len(reference))}
    cloud_tree, ref_tree = cKDTree(cloud), cKDTree(reference)
    c2r = cloud_tree.query(reference, k=1, workers=-1)[0]
    r2c = ref_tree.query(cloud, k=1, workers=-1)[0]
    return {"chamfer_m": float((c2r.mean() + r2c.mean()) / 2), "precision_at_5cm": float(np.mean(r2c <= tolerance)), "recall_at_5cm": float(np.mean(c2r <= tolerance)), "cloud_points": int(len(cloud)), "reference_points": int(len(reference))}


def run(args: argparse.Namespace) -> dict:
    root = Path(args.root).resolve()
    scenes = sorted(x for x in root.iterdir() if x.is_dir() and (x / "transforms_test.json").exists())
    prepared = [load_scene(scene, max_frames=args.max_frames, stride=args.stride) for scene in scenes]
    threshold = scene_adaptive_threshold([r["invalid_ratio"] for x in prepared[:2] for r in x["records"]], quantile=args.quantile)
    rows = []
    for index, item in enumerate(prepared):
        ref = _reference(item, args.max_points, args.seed + index)
        for variant in ("baseline", "m1", "m3", "full"):
            row = {"sequence": item["sequence"], "variant": variant, "threshold": threshold}
            row.update(_metrics(_cloud(item, variant, threshold), ref, args.tolerance))
            rows.append(row)
    payload = {"status": "PASS", "config": {"root": str(root), "max_frames": args.max_frames, "stride": args.stride, "heldout_rule": "every fourth sampled frame", "voxel_size_m": 0.03, "tolerance_m": args.tolerance, "max_reference_points": args.max_points, "threshold": threshold, "heldout_depth_used_only_for_evaluation": True}, "results": rows}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(rows), "output": str(args.output)}, indent=2))
    return payload


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--max-frames", type=int, default=48); p.add_argument("--stride", type=int, default=4); p.add_argument("--quantile", type=float, default=.9); p.add_argument("--tolerance", type=float, default=.05); p.add_argument("--max-points", type=int, default=12000); p.add_argument("--seed", type=int, default=31); run(p.parse_args())
