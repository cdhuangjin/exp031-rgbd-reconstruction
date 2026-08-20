"""Reproducible, lightweight RGB-D fusion experiment on TUM archives.

This is an explicit point-fusion baseline, not a learned NeRF/3DGS claim. It
is intended to make the three reliability modules measurable on real data
without silently changing the reconstruction backbone.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rgbd_fusion import depth_error_metrics, depth_to_camera_points, select_frame_indices, voxel_fuse, voxel_fuse_torch
from small_modules import pose_frame_weight, scene_adaptive_threshold


INTRINSICS = {
    "freiburg1": (517.3, 516.5, 318.6, 255.3),
    "freiburg2": (520.9, 521.0, 325.1, 249.7),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
}


def _lines_from_archive(archive: Path) -> tuple[list[tuple[float, str]], list[tuple[float, str]], list[tuple[float, np.ndarray]]]:
    rgb_lines: list[tuple[float, str]] = []
    depth_lines: list[tuple[float, str]] = []
    poses: list[tuple[float, np.ndarray]] = []
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            name = member.name.lstrip("./")
            if name.endswith("rgb.txt") or name.endswith("depth.txt") or name.endswith("groundtruth.txt"):
                stream = tf.extractfile(member)
                if stream is None:
                    continue
                for raw in stream.read().decode("utf-8", errors="ignore").splitlines():
                    raw = raw.strip()
                    if not raw or raw.startswith("#"):
                        continue
                    fields = raw.split()
                    if name.endswith("rgb.txt") and len(fields) >= 2:
                        rgb_lines.append((float(fields[0]), fields[1]))
                    elif name.endswith("depth.txt") and len(fields) >= 2:
                        depth_lines.append((float(fields[0]), fields[1]))
                    elif name.endswith("groundtruth.txt") and len(fields) >= 8:
                        poses.append((float(fields[0]), np.asarray([float(v) for v in fields[1:8]])))
    return rgb_lines, depth_lines, poses


def _nearest_pose(timestamp: float, poses: list[tuple[float, np.ndarray]]) -> np.ndarray:
    if not poses:
        return np.eye(4)
    ts, value = min(poses, key=lambda item: abs(item[0] - timestamp))
    x, y, z, qx, qy, qz, qw = value
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    rotation = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = [x, y, z]
    return pose


def _read_selected_depths(archive: Path, selected_names: set[str]) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            name = member.name.lstrip("./")
            matches = [selected for selected in selected_names if name == selected or name.endswith("/" + selected)]
            if not matches:
                continue
            stream = tf.extractfile(member)
            if stream is not None:
                with Image.open(io.BytesIO(stream.read())) as image:
                    images[matches[0]] = np.asarray(image.convert("I"), dtype=np.float32) / 5000.0
    return images


def _sequence_name(archive: Path) -> str:
    return archive.name.replace("rgbd_dataset_", "").replace(".tgz", "")


def _family(name: str) -> str:
    for prefix in INTRINSICS:
        if name.startswith(prefix):
            return prefix
    return "freiburg1"


def prepare_sequence(archive: Path, *, max_frames: int, stride: int) -> dict[str, Any]:
    name = _sequence_name(archive)
    rgb_lines, depth_lines, poses = _lines_from_archive(archive)
    indices = select_frame_indices(len(rgb_lines), stride=stride, max_frames=max_frames)
    selected = []
    for i in indices:
        timestamp, rgb_path = rgb_lines[i]
        depth_timestamp, depth_path = min(depth_lines, key=lambda item: abs(item[0] - timestamp))
        # TUM RGB and depth streams can have a fixed sensor-clock offset;
        # nearest-neighbour pairing at 1 s is the dataset convention here.
        if abs(depth_timestamp - timestamp) <= 1.0:
            selected.append((timestamp, rgb_path, depth_path))
    depth_names = {path for _, _, path in selected}
    depths = _read_selected_depths(archive, depth_names)
    fx, fy, cx, cy = INTRINSICS[_family(name)]
    records: list[dict[str, Any]] = []
    previous_pose = None
    for timestamp, _, depth_path in selected:
        depth = depths.get(depth_path)
        if depth is None:
            continue
        valid = np.isfinite(depth) & (depth > 0.01) & (depth < 10.0)
        invalid_ratio = 1.0 - float(valid.mean())
        pose = _nearest_pose(timestamp, poses)
        translation_delta = 0.0 if previous_pose is None else float(np.linalg.norm(pose[:3, 3] - previous_pose[:3, 3]))
        if previous_pose is None:
            rotation_delta = 0.0
        else:
            cosine = np.clip((np.trace(pose[:3, :3] @ previous_pose[:3, :3].T) - 1) / 2, -1, 1)
            rotation_delta = math.degrees(math.acos(float(cosine)))
        previous_pose = pose

        points, pixels = depth_to_camera_points(
            depth, fx=fx, fy=fy, cx=cx, cy=cy, depth_min=0.01, depth_max=10.0
        )
        world = (pose[:3, :3] @ points.T).T + pose[:3, 3]
        records.append(
            {
                "points": world,
                "pixels": pixels,
                "depth": depth,
                "pose": pose,
                "intrinsics": (fx, fy, cx, cy),
                "invalid_ratio": invalid_ratio,
                "translation_delta": translation_delta,
                "rotation_delta": rotation_delta,
            }
        )
    return {"sequence": name, "frames_requested": len(indices), "records": records}


def _project_world_points(points: np.ndarray, record: dict[str, Any], *, depth_min: float, depth_max: float) -> np.ndarray:
    depth = record["depth"]
    # During robustness experiments, ``pose`` may be perturbed for fusion
    # while ``eval_pose`` preserves the reference trajectory for evaluation.
    pose = record.get("eval_pose", record["pose"])
    fx, fy, cx, cy = record["intrinsics"]
    camera = (pose[:3, :3].T @ (points - pose[:3, 3]).T).T
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] >= depth_min) & (camera[:, 2] <= depth_max)
    camera = camera[valid]
    output = np.full(depth.shape, np.nan, dtype=np.float64)
    if len(camera) == 0:
        return output
    cols = np.rint(fx * camera[:, 0] / camera[:, 2] + cx).astype(np.int64)
    rows = np.rint(fy * camera[:, 1] / camera[:, 2] + cy).astype(np.int64)
    inside = (rows >= 0) & (rows < depth.shape[0]) & (cols >= 0) & (cols < depth.shape[1])
    rows, cols, z = rows[inside], cols[inside], camera[:, 2][inside]
    flat = np.full(depth.size, np.inf, dtype=np.float64)
    np.minimum.at(flat, rows * depth.shape[1] + cols, z)
    flat[~np.isfinite(flat)] = np.nan
    return flat.reshape(depth.shape)


def run_prepared(prepared: dict[str, Any], *, variant: str, threshold: float, device: str = "cpu") -> dict[str, Any]:
    start = time.perf_counter()
    all_points: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    frame_stats: list[dict[str, float]] = []
    for record in prepared["records"]:
        invalid_ratio = float(record["invalid_ratio"])
        if variant in {"m3", "full"} and invalid_ratio > threshold:
            frame_stats.append({"invalid_ratio": invalid_ratio, "weight": 0.0})
            continue
        points = record["points"]
        if variant in {"m1", "full", "m3"}:
            points = points[(points[:, 2] > 0.30) & (points[:, 2] < 5.0)]
        weight = pose_frame_weight(
            float(record["translation_delta"]), float(record["rotation_delta"]), invalid_ratio
        ) if variant in {"m2", "full"} else 1.0
        all_points.append(points)
        all_weights.append(np.full(len(points), weight, dtype=np.float64))
        frame_stats.append({"invalid_ratio": invalid_ratio, "weight": weight, "points": float(len(points))})

    points = np.concatenate(all_points) if all_points else np.empty((0, 3))
    weights = np.concatenate(all_weights) if all_weights else np.empty((0,))
    fused = voxel_fuse(points, weights, voxel_size=0.03) if device == "cpu" else voxel_fuse_torch(
        points, weights, voxel_size=0.03, device=device
    )
    # Held-out evaluation: build from 3/4 of the sampled frames and render
    # into the remaining frames. The held-out depth is never fused.
    build_points: list[np.ndarray] = []
    build_weights: list[np.ndarray] = []
    for index, record in enumerate(prepared["records"]):
        if index % 4 == 0 or (variant in {"m3", "full"} and float(record["invalid_ratio"]) > threshold):
            continue
        candidate = record["points"]
        if variant in {"m1", "full", "m3"}:
            candidate = candidate[(candidate[:, 2] > 0.30) & (candidate[:, 2] < 5.0)]
        build_points.append(candidate)
        frame_weight = pose_frame_weight(
            float(record["translation_delta"]), float(record["rotation_delta"]), float(record["invalid_ratio"])
        ) if variant in {"m2", "full"} else 1.0
        build_weights.append(np.full(len(candidate), frame_weight, dtype=np.float64))
    build_raw = np.concatenate(build_points) if build_points else np.empty((0, 3))
    build_raw_weights = np.concatenate(build_weights) if build_weights else np.empty((0,))
    build_fused = (voxel_fuse(build_raw, build_raw_weights, voxel_size=0.03) if device == "cpu" else voxel_fuse_torch(
        build_raw, build_raw_weights, voxel_size=0.03, device=device
    ))
    build_cloud = np.asarray(build_fused["weighted_centroid"], dtype=np.float64)
    heldout_rows = []
    for index, record in enumerate(prepared["records"]):
        if index % 4 != 0 or len(build_cloud) == 0:
            continue
        predicted = _project_world_points(build_cloud, record, depth_min=0.30, depth_max=5.0)
        heldout_rows.append(depth_error_metrics(predicted, record["depth"], depth_min=0.30, depth_max=5.0))
    valid_rows = [row for row in heldout_rows if np.isfinite(row["rmse"])]
    heldout_rmse = float(np.mean([row["rmse"] for row in valid_rows])) if valid_rows else float("nan")
    heldout_mae = float(np.mean([row["mae"] for row in valid_rows])) if valid_rows else float("nan")
    heldout_coverage = float(np.mean([row["coverage"] for row in valid_rows])) if valid_rows else 0.0
    return {
        "sequence": prepared["sequence"],
        "variant": variant,
        "frames_requested": prepared["frames_requested"],
        "frames_used": len(all_points),
        "raw_points": int(len(points)),
        "voxel_count": fused["voxel_count"],
        "coverage_proxy": fused["coverage"],
        "mean_frame_weight": float(np.mean([s["weight"] for s in frame_stats])) if frame_stats else 0.0,
        "heldout_frames": len(valid_rows),
        "heldout_depth_rmse": heldout_rmse,
        "heldout_depth_mae": heldout_mae,
        "heldout_depth_coverage": heldout_coverage,
        "threshold": threshold,
        "device": device,
        "wall_time_seconds": time.perf_counter() - start,
    }


def run_sequence(
    archive: Path, *, variant: str, max_frames: int, stride: int, threshold: float, device: str = "cpu"
) -> dict[str, Any]:
    return run_prepared(
        prepare_sequence(archive, max_frames=max_frames, stride=stride),
        variant=variant,
        threshold=threshold,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tum-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    archives = sorted(args.tum_root.glob("rgbd_dataset_*.tgz"))
    if not archives:
        raise SystemExit("no TUM archives found")
    variants = ["baseline", "m1", "m2", "m3", "full"]
    prepared = [prepare_sequence(archive, max_frames=args.max_frames, stride=args.stride) for archive in archives]
    source_errors = [
        float(record["invalid_ratio"])
        for item in prepared[:2]
        for record in item["records"]
    ]
    threshold = scene_adaptive_threshold(source_errors, quantile=0.9)
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is unavailable")
    results = [
        run_prepared(item, variant=variant, threshold=threshold, device=args.device)
        for item in prepared
        for variant in variants
    ]
    payload = {"config": vars(args) | {"threshold_source": "freiburg1_desk+desk2", "threshold": threshold}, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
