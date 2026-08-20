"""Run a small, explicit sparse-TSDF baseline on the TUM RGB-D archives.

This is deliberately a classical reference implementation rather than a
claim of a production Open3D/BundleFusion system.  It integrates signed
distance samples in a sparse voxel volume and extracts the near-zero surface
band for the same held-out depth evaluation used by the reliability study.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_candidates = [
    Path(__file__).resolve().parents[1] / "02_代码",
    Path(__file__).resolve().parents[1] / "code",
    Path(__file__).resolve().parent,
]
for CODE_DIR in _candidates:
    if (CODE_DIR / "rgbd_fusion.py").exists() and str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
        break

from rgbd_fusion import depth_error_metrics  # noqa: E402
from run_rgbd_fusion import (  # noqa: E402
    _project_world_points,
    prepare_sequence,
)


def sparse_tsdf_surface(
    records: list[dict],
    build_indices: list[int],
    *,
    voxel_size: float = 0.03,
    truncation: float = 0.06,
    pixel_stride: int = 4,
    surface_band: float = 0.25,
    device: str = "cpu",
) -> tuple[np.ndarray, dict[str, float]]:
    """Integrate a sparse TSDF and return voxel centres near its zero crossing."""

    if voxel_size <= 0 or truncation <= 0 or pixel_stride <= 0:
        raise ValueError("voxel_size, truncation, and pixel_stride must be positive")
    world_samples: list[np.ndarray] = []
    sdf_samples: list[np.ndarray] = []
    samples = 0
    for index in build_indices:
        record = records[index]
        depth = np.asarray(record["depth"], dtype=np.float64)
        fx, fy, cx, cy = record["intrinsics"]
        rows, cols = np.indices(depth.shape)
        keep = (
            (rows % pixel_stride == 0)
            & (cols % pixel_stride == 0)
            & np.isfinite(depth)
            & (depth >= 0.30)
            & (depth <= 5.0)
        )
        rows = rows[keep].astype(np.float64)
        cols = cols[keep].astype(np.float64)
        surface_z = depth[keep]
        if len(surface_z) == 0:
            continue
        rotation = np.asarray(record["pose"][:3, :3], dtype=np.float64)
        translation = np.asarray(record["pose"][:3, 3], dtype=np.float64)
        for offset in np.linspace(-truncation, truncation, 5):
            z = np.clip(surface_z + offset, 0.30, 5.0)
            camera = np.column_stack(
                ((cols - cx) * z / fx, (rows - cy) * z / fy, z)
            )
            world = (rotation @ camera.T).T + translation
            signed_distance = np.clip((surface_z - z) / truncation, -1.0, 1.0)
            world_samples.append(world)
            sdf_samples.append(signed_distance)
            samples += len(surface_z)

    if not world_samples:
        return np.empty((0, 3), dtype=np.float64), {"tsdf_voxels": 0, "samples": samples}
    points = np.concatenate(world_samples, axis=0)
    signed = np.concatenate(sdf_samples, axis=0)
    if device == "cuda":
        import torch

        point_tensor = torch.as_tensor(points, dtype=torch.float64, device="cuda")
        sdf_tensor = torch.as_tensor(signed, dtype=torch.float64, device="cuda")
        keys_tensor = torch.floor(point_tensor / float(voxel_size)).to(torch.int64)
        keys, inverse = torch.unique(keys_tensor, dim=0, return_inverse=True)
        sums = torch.zeros(len(keys), dtype=torch.float64, device="cuda")
        counts_tensor = torch.zeros(len(keys), dtype=torch.float64, device="cuda")
        sums.scatter_add_(0, inverse, sdf_tensor)
        counts_tensor.scatter_add_(0, inverse, torch.ones_like(sdf_tensor))
        values = sums / counts_tensor.clamp_min(1.0)
        selected = torch.abs(values) <= float(surface_band)
        surface = ((keys[selected].to(torch.float64) + 0.5) * float(voxel_size)).cpu().numpy()
        voxel_count = int(len(keys))
    else:
        keys, inverse = np.unique(np.floor(points / voxel_size).astype(np.int64), axis=0, return_inverse=True)
        sums = np.bincount(inverse, weights=signed, minlength=len(keys))
        counts = np.bincount(inverse, minlength=len(keys))
        values = sums / np.maximum(counts, 1)
        selected = np.abs(values) <= surface_band
        surface = (keys[selected].astype(np.float64) + 0.5) * voxel_size
        voxel_count = int(len(keys))
    return surface, {
        "tsdf_voxels": voxel_count,
        "surface_voxels": int(len(surface)),
        "samples": int(samples),
    }


def run_sequence(archive: Path, *, max_frames: int, stride: int, device: str, pixel_stride: int) -> dict:
    start = time.perf_counter()
    prepared = prepare_sequence(archive, max_frames=max_frames, stride=stride)
    records = prepared["records"]
    build_indices = [i for i in range(len(records)) if i % 4 != 0]
    heldout_indices = [i for i in range(len(records)) if i % 4 == 0]
    surface, stats = sparse_tsdf_surface(records, build_indices, device=device, pixel_stride=pixel_stride)
    rows = []
    for index in heldout_indices:
        if len(surface) == 0:
            continue
        predicted = _project_world_points(surface, records[index], depth_min=0.30, depth_max=5.0)
        rows.append(depth_error_metrics(predicted, records[index]["depth"], depth_min=0.30, depth_max=5.0))
    valid = [row for row in rows if np.isfinite(row["rmse"])]
    return {
        "sequence": prepared["sequence"],
        "variant": "sparse_tsdf",
        "device": device,
        "frames_requested": prepared["frames_requested"],
        "build_frames": len(build_indices),
        "heldout_frames": len(valid),
        "heldout_depth_rmse": float(np.mean([row["rmse"] for row in valid])) if valid else float("nan"),
        "heldout_depth_mae": float(np.mean([row["mae"] for row in valid])) if valid else float("nan"),
        "heldout_depth_coverage": float(np.mean([row["coverage"] for row in valid])) if valid else 0.0,
        **stats,
        "wall_time_seconds": time.perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--stride", type=int, default=80)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pixel-stride", type=int, default=4)
    args = parser.parse_args()
    archives = sorted(args.data_dir.glob("rgbd_dataset_*.tgz"))
    results = [
        run_sequence(
            path,
            max_frames=args.max_frames,
            stride=args.stride,
            device=args.device,
            pixel_stride=args.pixel_stride,
        )
        for path in archives
    ]
    payload = {
        "config": {
            "max_frames": args.max_frames,
            "stride": args.stride,
            "voxel_size": 0.03,
            "truncation": 0.06,
            "pixel_stride": args.pixel_stride,
            "device": args.device,
        },
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
