from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class CandidateView:
    view_id: str
    nominal_gain: float
    pose_gain_std: float
    uncovered_risk: float
    cost: float


def perturb_pose_errors(
    count: int,
    translation_std: float,
    rotation_std_deg: float,
    seed: int,
) -> list[tuple[float, float, float, float, float, float]]:
    if count < 1:
        raise ValueError("count must be positive")
    if translation_std < 0 or rotation_std_deg < 0:
        raise ValueError("pose standard deviations must be nonnegative")

    generator = random.Random(seed)
    return [
        tuple(
            generator.gauss(0.0, translation_std if axis < 3 else rotation_std_deg)
            for axis in range(6)
        )
        for _ in range(count)
    ]


def _utility(
    candidate: CandidateView,
    mode: str,
    lambda_pose: float,
    lambda_cost: float,
    lambda_cov: float,
) -> float:
    if mode == "nominal":
        return candidate.nominal_gain
    if mode == "pose_robust":
        return candidate.nominal_gain - lambda_pose * candidate.pose_gain_std
    if mode == "full_risk":
        return (
            candidate.nominal_gain
            - lambda_pose * candidate.pose_gain_std
            - lambda_cost * candidate.cost
            - lambda_cov * candidate.uncovered_risk
        )
    raise ValueError(f"unknown mode: {mode}")


def select_next_view(
    candidates: list[CandidateView],
    *,
    mode: str,
    lambda_pose: float = 1.0,
    lambda_cost: float = 0.1,
    lambda_cov: float = 0.5,
) -> CandidateView:
    if not candidates:
        raise ValueError("candidates must not be empty")
    if min(lambda_pose, lambda_cost, lambda_cov) < 0:
        raise ValueError("utility weights must be nonnegative")

    ranked = sorted(
        candidates,
        key=lambda item: (
            -_utility(item, mode, lambda_pose, lambda_cost, lambda_cov),
            item.view_id,
        ),
    )
    return ranked[0]
