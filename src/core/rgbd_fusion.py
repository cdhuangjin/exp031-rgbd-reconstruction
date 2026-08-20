"""Small, deterministic RGB-D point-fusion primitives.

The module deliberately contains no dataset-specific policy.  The runner is
responsible for reading TUM archives and deciding which reliability variant
to use; these functions only implement the geometry used by every variant.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def select_frame_indices(total: int, *, stride: int = 1, max_frames: int | None = None) -> list[int]:
    if total < 0 or stride < 1:
        raise ValueError("total must be non-negative and stride must be positive")
    indices = list(range(0, total, stride))
    if max_frames is not None and total <= max_frames:
        indices = list(range(total))
    if total and (not indices or indices[-1] != total - 1):
        indices.append(total - 1)
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        if len(indices) > max_frames:
            positions = np.linspace(0, len(indices) - 1, max_frames).round().astype(int)
            indices = [indices[int(position)] for position in positions]
    return list(dict.fromkeys(indices))


def depth_to_camera_points(
    depth: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_min: float,
    depth_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    if depth.ndim != 2 or fx <= 0 or fy <= 0 or depth_min < 0 or depth_max <= depth_min:
        raise ValueError("invalid depth image or camera/depth bounds")
    valid = np.isfinite(depth) & (depth >= depth_min) & (depth <= depth_max)
    rows, cols = np.nonzero(valid)
    z = depth[rows, cols].astype(np.float64)
    x = (cols.astype(np.float64) - cx) * z / fx
    y = (rows.astype(np.float64) - cy) * z / fy
    return np.column_stack((x, y, z)), np.column_stack((rows, cols))


def depth_error_metrics(
    predicted: np.ndarray, ground_truth: np.ndarray, *, depth_min: float, depth_max: float
) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    if predicted.shape != ground_truth.shape or predicted.ndim != 2:
        raise ValueError("predicted and ground_truth must be equally shaped depth images")
    valid = (
        np.isfinite(predicted)
        & np.isfinite(ground_truth)
        & (predicted >= depth_min)
        & (predicted <= depth_max)
        & (ground_truth >= depth_min)
        & (ground_truth <= depth_max)
    )
    total = valid.size
    count = int(valid.sum())
    if count == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "coverage": 0.0, "valid_pixels": 0.0}
    error = predicted[valid] - ground_truth[valid]
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "coverage": count / total,
        "valid_pixels": float(count),
    }


def voxel_fuse(points: np.ndarray, weights: np.ndarray, *, voxel_size: float) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or weights.shape != (len(points),):
        raise ValueError("points must be [N,3] and weights must be [N]")
    if voxel_size <= 0 or np.any(~np.isfinite(points)) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("invalid points, weights, or voxel size")
    if len(points) == 0:
        return {"voxel_count": 0, "weighted_centroid": [], "coverage": 0.0}

    buckets: dict[tuple[int, int, int], list[np.ndarray | float]] = defaultdict(lambda: [np.zeros(3), 0.0])
    for point, weight in zip(points, weights):
        key = tuple(np.floor(point / voxel_size).astype(int))
        buckets[key][0] += point * weight
        buckets[key][1] += float(weight)

    centroids = []
    for weighted_sum, total_weight in buckets.values():
        if total_weight > 0:
            centroids.append((weighted_sum / total_weight).tolist())
    return {
        "voxel_count": len(buckets),
        "weighted_centroid": centroids,
        "coverage": len(buckets) / len(points),
    }


def voxel_fuse_torch(
    points: np.ndarray, weights: np.ndarray, *, voxel_size: float, device: str = "cuda"
) -> dict[str, Any]:
    """GPU-capable equivalent of :func:`voxel_fuse` using real tensor ops."""
    import torch

    # Keep float64 to make voxel-boundary decisions agree with the NumPy
    # reference; float32 can move a point across a 3 cm voxel boundary.
    points_np = np.asarray(points, dtype=np.float64)
    weights_np = np.asarray(weights, dtype=np.float64)
    if points_np.ndim != 2 or points_np.shape[1] != 3 or weights_np.shape != (len(points_np),):
        raise ValueError("points must be [N,3] and weights must be [N]")
    if voxel_size <= 0:
        raise ValueError("voxel size must be positive")
    if len(points_np) == 0:
        return {"voxel_count": 0, "coverage": 0.0, "weighted_centroid": []}
    point_tensor = torch.as_tensor(points_np, device=device)
    weight_tensor = torch.as_tensor(weights_np, device=device)
    keys = torch.floor(point_tensor / float(voxel_size)).to(torch.int64)
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    voxel_count = int(inverse.max().item()) + 1
    weighted_sum = torch.zeros((voxel_count, 3), device=device, dtype=point_tensor.dtype)
    weight_sum = torch.zeros((voxel_count,), device=device, dtype=weight_tensor.dtype)
    weighted_sum.index_add_(0, inverse, point_tensor * weight_tensor[:, None])
    weight_sum.index_add_(0, inverse, weight_tensor)
    centroids = (weighted_sum / weight_sum.clamp_min(1e-12)[:, None]).detach().cpu().numpy().tolist()
    return {"voxel_count": voxel_count, "coverage": voxel_count / len(points_np), "weighted_centroid": centroids}
