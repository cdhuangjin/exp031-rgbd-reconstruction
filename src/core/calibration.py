from __future__ import annotations

import math


def calibration_radius(residuals: list[float], *, alpha: float) -> float:
    if not residuals:
        raise ValueError("residuals must not be empty")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    values = sorted(abs(float(value)) for value in residuals)
    rank = min(len(values) - 1, math.ceil((len(values) + 1) * (1 - alpha)) - 1)
    return values[rank]


def should_stop(
    *,
    predicted_quality: float,
    radius: float,
    target_quality: float,
    uncovered_risk: float,
    max_risk: float,
) -> bool:
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if not 0 <= uncovered_risk <= 1 or not 0 <= max_risk <= 1:
        raise ValueError("risk values must be in [0, 1]")
    return predicted_quality - radius >= target_quality and uncovered_risk <= max_risk
