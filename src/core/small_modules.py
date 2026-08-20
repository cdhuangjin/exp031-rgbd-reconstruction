from __future__ import annotations

import math
from typing import Sequence


def depth_validity_weights(
    depths: Sequence[float], *, min_depth: float, max_depth: float
) -> list[float]:
    if min_depth < 0 or max_depth <= min_depth:
        raise ValueError("depth range is invalid")
    weights: list[float] = []
    for depth in depths:
        value = float(depth)
        weights.append(1.0 if min_depth <= value <= max_depth else 0.0)
    return weights


def pose_frame_weight(
    translation_delta: float,
    rotation_delta_deg: float,
    reprojection_error: float,
    *,
    lambda_translation: float = 1.0,
    lambda_rotation: float = 0.05,
    lambda_reprojection: float = 1.0,
) -> float:
    values = (
        translation_delta,
        rotation_delta_deg,
        reprojection_error,
        lambda_translation,
        lambda_rotation,
        lambda_reprojection,
    )
    if any(float(value) < 0 for value in values):
        raise ValueError("pose statistics and weights must be nonnegative")
    penalty = (
        lambda_translation * translation_delta
        + lambda_rotation * rotation_delta_deg
        + lambda_reprojection * reprojection_error
    )
    return math.exp(-penalty)


def scene_adaptive_threshold(errors: Sequence[float], *, quantile: float) -> float:
    if not errors:
        raise ValueError("errors must not be empty")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    values = sorted(float(error) for error in errors)
    if values[0] < 0:
        raise ValueError("errors must be nonnegative")
    index = min(len(values) - 1, int((len(values) - 1) * quantile))
    return values[index]


def combine_frame_reliability(depth_weight: float, pose_weight: float) -> float:
    if not 0 <= depth_weight <= 1 or not 0 <= pose_weight <= 1:
        raise ValueError("reliability weights must be in [0, 1]")
    return depth_weight * pose_weight
