"""Cross-dataset RGB-D fusion evaluation on the uploaded NeRF Synthetic renders."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

CODE_ROOT = Path(__file__).resolve().parents[1] / "02_代码"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from rgbd_fusion import depth_error_metrics, select_frame_indices, voxel_fuse_torch
from small_modules import pose_frame_weight, scene_adaptive_threshold


def decode_depth(path: Path, *, scale: float = 4.0) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return image / 255.0 * scale


def _camera_to_world(matrix: list[list[float]], depth: np.ndarray, *, focal: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    rows, cols = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 0.02) & (depth < 5.0)
    z = -depth[valid]
    x = (cols[valid] - (width - 1) / 2.0) * depth[valid] / focal
    y = -((rows[valid] - (height - 1) / 2.0) * depth[valid] / focal)
    camera = np.column_stack([x, y, z])
    pose = np.asarray(matrix, dtype=np.float64)
    world = (pose[:3, :3] @ camera.T).T + pose[:3, 3]
    pixels = np.column_stack([rows[valid], cols[valid]])
    return world, pixels


def _project(points: np.ndarray, record: dict, *, depth_min: float = 0.02, depth_max: float = 5.0) -> np.ndarray:
    pose = np.asarray(record["pose"], dtype=np.float64)
    camera = (pose[:3, :3].T @ (points - pose[:3, 3]).T).T
    depth = -camera[:, 2]
    valid = np.isfinite(camera).all(axis=1) & (depth >= depth_min) & (depth <= depth_max)
    camera, depth = camera[valid], depth[valid]
    output = np.full(record["depth"].shape, np.nan, dtype=np.float64)
    if len(camera) == 0:
        return output
    h, w = output.shape
    cols = np.rint(record["focal"] * camera[:, 0] / depth + (w - 1) / 2.0).astype(int)
    rows = np.rint(-(record["focal"] * camera[:, 1] / depth) + (h - 1) / 2.0).astype(int)
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    flat = np.full(output.size, np.inf, dtype=np.float64)
    np.minimum.at(flat, rows[inside] * w + cols[inside], depth[inside])
    flat[~np.isfinite(flat)] = np.nan
    return flat.reshape(output.shape)


def load_scene(scene_dir: Path, *, max_frames: int, stride: int) -> dict:
    metadata = json.loads((scene_dir / "transforms_test.json").read_text(encoding="utf-8"))
    frames = metadata["frames"]
    indices = select_frame_indices(len(frames), stride=stride, max_frames=max_frames)
    image_size = 800
    focal = 0.5 * image_size / math.tan(float(metadata["camera_angle_x"]) / 2.0)
    records = []
    previous = None
    for index in indices:
        frame = frames[index]
        stem = Path(frame["file_path"]).name
        depth_candidates = sorted((scene_dir / "test").glob(f"{stem}_depth_*.png"))
        if not depth_candidates:
            raise FileNotFoundError(f"no rendered depth for {stem} in {scene_dir}")
        depth = decode_depth(depth_candidates[0])
        pose = np.asarray(frame["transform_matrix"], dtype=np.float64)
        invalid_ratio = float(np.mean(depth <= 0.02))
        translation_delta = 0.0 if previous is None else float(np.linalg.norm(pose[:3, 3] - previous[:3, 3]))
        rotation_delta = 0.0 if previous is None else math.degrees(math.acos(float(np.clip((np.trace(pose[:3, :3] @ previous[:3, :3].T) - 1) / 2, -1, 1))))
        points, _ = _camera_to_world(pose.tolist(), depth, focal=focal)
        records.append({"depth": depth, "pose": pose, "points": points, "focal": focal, "invalid_ratio": invalid_ratio, "translation_delta": translation_delta, "rotation_delta": rotation_delta})
        previous = pose
    return {"sequence": scene_dir.name, "frames_requested": len(indices), "records": records}


def run_prepared(prepared: dict, *, variant: str, threshold: float) -> dict:
    start = time.perf_counter()
    build_points, weights = [], []
    for i, record in enumerate(prepared["records"]):
        if i % 4 == 0:
            continue
        if variant in {"m3", "full"} and record["invalid_ratio"] > threshold:
            continue
        points = record["points"]
        if variant in {"m1", "m3", "full"}:
            points = points[(points[:, 2] > -5.0) & (points[:, 2] < 5.0)]
        weight = pose_frame_weight(record["translation_delta"], record["rotation_delta"], record["invalid_ratio"]) if variant in {"m2", "full"} else 1.0
        build_points.append(points)
        weights.append(np.full(len(points), weight, dtype=np.float64))
    points = np.concatenate(build_points) if build_points else np.empty((0, 3))
    point_weights = np.concatenate(weights) if weights else np.empty((0,))
    fused = voxel_fuse_torch(points, point_weights, voxel_size=0.03, device="cuda")
    cloud = np.asarray(fused["weighted_centroid"], dtype=np.float64)
    metrics = []
    for i, record in enumerate(prepared["records"]):
        if i % 4 != 0 or len(cloud) == 0:
            continue
        metrics.append(depth_error_metrics(_project(cloud, record), record["depth"], depth_min=0.02, depth_max=5.0))
    valid = [x for x in metrics if np.isfinite(x["rmse"])]
    return {"sequence": prepared["sequence"], "variant": variant, "frames_requested": prepared["frames_requested"], "heldout_frames": len(valid), "heldout_depth_rmse": float(np.mean([x["rmse"] for x in valid])), "heldout_depth_mae": float(np.mean([x["mae"] for x in valid])), "heldout_depth_coverage": float(np.mean([x["coverage"] for x in valid])), "threshold": threshold, "device": "cuda", "wall_time_seconds": time.perf_counter() - start}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    scenes = sorted(x for x in args.root.iterdir() if x.is_dir() and (x / "transforms_test.json").exists())
    prepared = [load_scene(scene, max_frames=args.max_frames, stride=args.stride) for scene in scenes]
    source_errors = [r["invalid_ratio"] for item in prepared[:2] for r in item["records"]]
    threshold = scene_adaptive_threshold(source_errors, quantile=0.9)
    results = [run_prepared(item, variant=variant, threshold=threshold) for item in prepared for variant in ["baseline", "m1", "m2", "m3", "full"]]
    payload = {"config": vars(args) | {"depth_encoding": "grayscale_uint8_div255_times4m", "source_scenes": [x["sequence"] for x in prepared[:2]], "threshold": threshold}, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"rows": len(results), "scenes": len(prepared), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
