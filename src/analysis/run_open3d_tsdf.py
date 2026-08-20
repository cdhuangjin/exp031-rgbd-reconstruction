"""Run an official Open3D ScalableTSDFVolume reference on TUM RGB-D."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1] / "02_代码"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import open3d as o3d  # noqa: E402
from rgbd_fusion import depth_error_metrics  # noqa: E402
from run_rgbd_fusion import _project_world_points, prepare_sequence  # noqa: E402


def run_sequence(archive: Path, *, max_frames: int, stride: int) -> dict:
    start = time.perf_counter()
    prepared = prepare_sequence(archive, max_frames=max_frames, stride=stride)
    records = prepared["records"]
    build_indices = [i for i in range(len(records)) if i % 4 != 0]
    heldout_indices = [i for i in range(len(records)) if i % 4 == 0]
    if not build_indices:
        raise ValueError("no build frames")
    fx, fy, cx, cy = records[0]["intrinsics"]
    height, width = records[0]["depth"].shape
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.03,
        sdf_trunc=0.06,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )
    blank_color = o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8))
    for index in build_indices:
        depth = np.asarray(records[index]["depth"], dtype=np.float32)
        depth_image = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            blank_color,
            depth_image,
            depth_scale=1.0,
            depth_trunc=5.0,
            convert_rgb_to_intensity=False,
        )
        # TUM pose is world_T_camera; Open3D integrate expects camera_T_world.
        volume.integrate(rgbd, intrinsic, np.linalg.inv(records[index]["pose"]))
    cloud = volume.extract_point_cloud()
    points = np.asarray(cloud.points, dtype=np.float64)
    rows = []
    for index in heldout_indices:
        if len(points) == 0:
            continue
        predicted = _project_world_points(points, records[index], depth_min=0.30, depth_max=5.0)
        rows.append(depth_error_metrics(predicted, records[index]["depth"], depth_min=0.30, depth_max=5.0))
    valid = [row for row in rows if np.isfinite(row["rmse"])]
    return {
        "sequence": prepared["sequence"],
        "variant": "open3d_tsdf",
        "device": "cpu_open3d",
        "frames_requested": prepared["frames_requested"],
        "build_frames": len(build_indices),
        "heldout_frames": len(valid),
        "tsdf_points": int(len(points)),
        "heldout_depth_rmse": float(np.mean([row["rmse"] for row in valid])) if valid else float("nan"),
        "heldout_depth_mae": float(np.mean([row["mae"] for row in valid])) if valid else float("nan"),
        "heldout_depth_coverage": float(np.mean([row["coverage"] for row in valid])) if valid else 0.0,
        "wall_time_seconds": time.perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--stride", type=int, default=80)
    args = parser.parse_args()
    archives = sorted(args.data_dir.glob("rgbd_dataset_*.tgz"))
    results = [run_sequence(path, max_frames=args.max_frames, stride=args.stride) for path in archives]
    payload = {
        "config": {
            "max_frames": args.max_frames,
            "stride": args.stride,
            "voxel_length": 0.03,
            "sdf_trunc": 0.06,
            "open3d_version": o3d.__version__,
            "device": "cpu_open3d",
        },
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
