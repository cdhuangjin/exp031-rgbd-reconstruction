from __future__ import annotations

import json
from pathlib import Path
import tarfile
from typing import Iterable, Sequence


def inspect_blender_scene(scene_root: str | Path) -> dict[str, object]:
    root = Path(scene_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Blender scene directory not found: {root}")

    frame_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        transform_path = root / f"transforms_{split}.json"
        if not transform_path.is_file():
            raise FileNotFoundError(f"missing Blender transform file: {transform_path.name}")
        payload = json.loads(transform_path.read_text(encoding="utf-8"))
        frames = payload.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"{transform_path.name} must contain a non-empty frames list")
        frame_counts[split] = len(frames)

    return {"scene": root.name, "frame_counts": frame_counts}


def inspect_tum_archive(archive_path: str | Path) -> dict[str, object]:
    archive = Path(archive_path)
    if not archive.is_file():
        raise FileNotFoundError(f"TUM archive not found: {archive}")
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    return {
        "archive": archive.name,
        "members": len(names),
        "has_rgb_index": any(name.endswith("/rgb.txt") for name in names),
        "has_depth_index": any(name.endswith("/depth.txt") for name in names),
        "has_groundtruth": any(name.endswith("/groundtruth.txt") for name in names),
        "has_rgb_images": any("/rgb/" in name and name.lower().endswith(".png") for name in names),
        "has_depth_images": any("/depth/" in name and name.lower().endswith(".png") for name in names),
    }


def nearest_timestamp_pairs(
    left: Sequence[tuple[float, str]],
    right: Sequence[tuple[float, str]],
    *,
    tolerance: float,
) -> list[tuple[str, str]]:
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    pairs: list[tuple[str, str]] = []
    right_sorted = sorted(right)
    for left_time, left_path in sorted(left):
        if not right_sorted:
            break
        nearest_time, nearest_path = min(right_sorted, key=lambda item: abs(item[0] - left_time))
        if abs(nearest_time - left_time) <= tolerance:
            pairs.append((left_path, nearest_path))
    return pairs


def sample_candidate_indices(frame_count: int, *, initial_count: int, step: int) -> list[int]:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if not 1 <= initial_count <= frame_count:
        raise ValueError("initial_count must be within frame_count")
    if step < 1:
        raise ValueError("step must be positive")
    initial = list(range(initial_count))
    first_candidate = ((initial_count + step - 1) // step) * step
    candidates = list(range(first_candidate, frame_count, step))
    return initial + candidates
