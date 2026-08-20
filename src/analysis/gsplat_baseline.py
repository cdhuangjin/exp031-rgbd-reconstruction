"""Small, explicit gsplat baseline helpers used by the cloud experiment.

The module deliberately keeps dataset parsing separate from optimization so the
public gsplat baseline can be audited without relying on Nerfstudio's cache.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def blender_intrinsics(width: int, height: int, camera_angle_x: float) -> tuple[float, float, float, float]:
    """Return pinhole intrinsics for the standard Blender synthetic metadata."""
    fx = 0.5 * width / math.tan(0.5 * camera_angle_x)
    fy = fx
    return float(fx), float(fy), float(width) / 2.0, float(height) / 2.0


def normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Normalize quaternion rows, mapping zero rows to the identity rotation."""
    q = np.asarray(quaternions, dtype=np.float32).copy()
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    zero = norms[:, 0] < 1e-8
    q /= np.maximum(norms, 1e-8)
    q[zero] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q


def select_frames(frames: Sequence[dict[str, Any]], count: int, seed: int = 0) -> list[dict[str, Any]]:
    """Select a reproducible subset without changing the selected order."""
    if count <= 0:
        raise ValueError("count must be positive")
    if count >= len(frames):
        return list(frames)
    indices = np.random.RandomState(seed).choice(len(frames), count, replace=False)
    return [frames[int(i)] for i in indices]


def decode_blender_depth(encoded: np.ndarray, scale: float = 4.0) -> np.ndarray:
    """Decode the 8-bit metric-depth convention used by NeRF Synthetic."""
    return np.asarray(encoded, dtype=np.float32) / 255.0 * float(scale)


def blender_pose_to_viewmat(transform_matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert a Blender camera-to-world matrix to an OpenCV-style view matrix."""
    c2w = np.asarray(transform_matrix, dtype=np.float32).copy()
    # Nerfstudio's Blender parser applies the same OpenGL-to-OpenCV axis flip.
    c2w[:3, 1:3] *= -1.0
    return np.linalg.inv(c2w).astype(np.float32)
