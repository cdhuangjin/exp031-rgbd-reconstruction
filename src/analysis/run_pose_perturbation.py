"""Pose-noise robustness experiment for the RGB-D fusion backbone."""

from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1] / "02_代码"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from run_rgbd_fusion import INTRINSICS, prepare_sequence, run_prepared
from rgbd_fusion import depth_to_camera_points


def _axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) or 1.0)
    x, y, z = axis
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.asarray([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def perturb_prepared(
    prepared: dict,
    *,
    translation_std: float,
    rotation_std_deg: float,
    seed: int,
) -> dict:
    """Perturb poses used for fusion while retaining the true pose for scoring."""
    rng = np.random.default_rng(seed)
    output = deepcopy(prepared)
    previous = None
    for record in output["records"]:
        true_pose = np.asarray(record["pose"], dtype=np.float64).copy()
        noisy_pose = true_pose.copy()
        axis = rng.normal(size=3)
        angle = math.radians(float(rng.normal(0.0, rotation_std_deg)))
        noisy_pose[:3, :3] = _axis_angle(axis, angle) @ true_pose[:3, :3]
        noisy_pose[:3, 3] += rng.normal(0.0, translation_std, size=3)
        fx, fy, cx, cy = record["intrinsics"]
        camera_points, _ = depth_to_camera_points(
            record["depth"], fx=fx, fy=fy, cx=cx, cy=cy, depth_min=0.01, depth_max=10.0
        )
        world_points = (noisy_pose[:3, :3] @ camera_points.T).T + noisy_pose[:3, 3]
        record["eval_pose"] = true_pose
        record["pose"] = noisy_pose
        record["points"] = world_points
        record["translation_delta"] = 0.0 if previous is None else float(
            np.linalg.norm(noisy_pose[:3, 3] - previous[:3, 3])
        )
        if previous is None:
            record["rotation_delta"] = 0.0
        else:
            cosine = np.clip((np.trace(noisy_pose[:3, :3] @ previous[:3, :3].T) - 1) / 2, -1, 1)
            record["rotation_delta"] = math.degrees(math.acos(float(cosine)))
        previous = noisy_pose
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tum-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=31031)
    args = parser.parse_args()
    archives = sorted(args.tum_root.glob("rgbd_dataset_*.tgz"))
    if not archives:
        raise SystemExit("no TUM archives found")
    prepared = [prepare_sequence(a, max_frames=args.max_frames, stride=args.stride) for a in archives]
    source_errors = [float(r["invalid_ratio"]) for item in prepared[:2] for r in item["records"]]
    from small_modules import scene_adaptive_threshold

    threshold = scene_adaptive_threshold(source_errors, quantile=0.9)
    levels = [("clean", 0.0, 0.0), ("mild", 0.01, 1.0), ("moderate", 0.03, 3.0), ("severe", 0.05, 5.0)]
    variants = ["baseline", "m1", "m2", "m3", "full"]
    results = []
    for level, translation_std, rotation_std_deg in levels:
        for scene_index, item in enumerate(prepared):
            noisy = perturb_prepared(
                item,
                translation_std=translation_std,
                rotation_std_deg=rotation_std_deg,
                seed=args.seed + scene_index,
            )
            for variant in variants:
                row = run_prepared(noisy, variant=variant, threshold=threshold, device=args.device)
                row.update({
                    "noise_level": level,
                    "translation_std_m": translation_std,
                    "rotation_std_deg": rotation_std_deg,
                    "seed": args.seed + scene_index,
                })
                results.append(row)
    payload = {"config": vars(args) | {"threshold": threshold, "noise_levels": [x[0] for x in levels]}, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"rows": len(results), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
