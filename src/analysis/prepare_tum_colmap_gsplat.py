"""Prepare a leakage-controlled COLMAP text capture for official gsplat.

The capture uses TUM RGB frames, calibrated intrinsics, and ground-truth
poses.  3-D initialization points are generated only from frames that are
not assigned to the every-Nth-image test split used by the official trainer.
This adapter is for a standard-gsplat smoke/benchmark and is not part of the
main RGB-D fusion method.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


INTRINSICS = {
    "freiburg1": (517.3, 516.5, 318.6, 255.3),
    "freiburg2": (520.9, 521.0, 325.1, 249.7),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
}


def lines(path: Path) -> list[tuple[float, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 2:
            out.append((float(p[0]), p[1]))
    return out


def nearest(ts: float, rows: list[tuple[float, str]], tol: float) -> str | None:
    if not rows:
        return None
    i = min(range(len(rows)), key=lambda j: abs(rows[j][0] - ts))
    return rows[i][1] if abs(rows[i][0] - ts) <= tol else None


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def rot_to_quat(r: np.ndarray) -> tuple[float, float, float, float]:
    # Returns COLMAP's qw, qx, qy, qz order.
    tr = float(np.trace(r))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        qw, qx, qy, qz = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        qw, qx, qy, qz = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        qw, qx, qy, qz = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    return tuple(float(x) for x in q)


def associate(root: Path) -> list[dict]:
    rgb, depth = lines(root / "rgb.txt"), lines(root / "depth.txt")
    gt = []
    for line in (root / "groundtruth.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) == 8:
            gt.append((float(p[0]), np.asarray([float(x) for x in p[1:]], dtype=np.float64)))
    rows = []
    for ts, rgb_rel in rgb:
        depth_rel = nearest(ts, depth, 0.025)
        if depth_rel is None or not gt:
            continue
        j = min(range(len(gt)), key=lambda i: abs(gt[i][0] - ts))
        if abs(gt[j][0] - ts) > 0.05:
            continue
        tx, ty, tz, qx, qy, qz, qw = gt[j][1]
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = quat_to_rot(qx, qy, qz, qw)
        c2w[:3, 3] = [tx, ty, tz]
        rows.append({"rgb": rgb_rel, "depth": depth_rel, "c2w": c2w})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--frames", type=int, default=50)
    p.add_argument("--test-every", type=int, default=8)
    p.add_argument("--factor", type=int, default=4)
    p.add_argument("--max-points-per-frame", type=int, default=600)
    args = p.parse_args()
    root, out = Path(args.data).resolve(), Path(args.output).resolve()
    family = next((k for k in INTRINSICS if k in root.name), "freiburg1")
    fx, fy, cx, cy = INTRINSICS[family]
    rows = associate(root)
    if len(rows) < args.frames:
        raise RuntimeError(f"only {len(rows)} matched frames, need {args.frames}")
    idx = np.linspace(0, len(rows) - 1, args.frames).round().astype(int)
    rows = [rows[int(i)] for i in idx]
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / f"images_{args.factor}").mkdir(parents=True, exist_ok=True)
    (out / "sparse/0").mkdir(parents=True, exist_ok=True)
    cam_w, cam_h = 640, 480
    for row in rows:
        src = root / row["rgb"]
        dst = out / "images" / Path(row["rgb"]).name
        shutil.copy2(src, dst)
        im = Image.open(src).convert("RGB")
        im.resize((cam_w // args.factor, cam_h // args.factor), Image.Resampling.BILINEAR).save(out / f"images_{args.factor}" / dst.name)

    (out / "sparse/0/cameras.txt").write_text(
        f"# Camera list\n# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS\n1 PINHOLE {cam_w} {cam_h} {fx:.9f} {fy:.9f} {cx:.9f} {cy:.9f}\n",
        encoding="utf-8",
    )
    image_lines = ["# Image list with two lines of data per image:", "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME"]
    for image_id, row in enumerate(rows, 1):
        w2c = np.linalg.inv(row["c2w"])
        qw, qx, qy, qz = rot_to_quat(w2c[:3, :3])
        tx, ty, tz = w2c[:3, 3]
        name = Path(row["rgb"]).name
        image_lines.append(f"{image_id} {qw:.12g} {qx:.12g} {qy:.12g} {qz:.12g} {tx:.12g} {ty:.12g} {tz:.12g} 1 {name}")
        image_lines.append("")
    (out / "sparse/0/images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")

    # Sparse colored points from training images only; every Nth image is held out.
    rng = np.random.default_rng(31)
    point_lines = ["# 3D point list with one line of data per point:", "# POINT3D_ID, X, Y, Z, R, G, B, ERROR"]
    pid = 1
    for i, row in enumerate(rows):
        if i % args.test_every == 0:
            continue
        depth = np.asarray(Image.open(root / row["depth"]), dtype=np.float32) / 5000.0
        rgb = np.asarray(Image.open(root / row["rgb"]).convert("RGB"))
        yy, xx = np.where((depth > 0.2) & (depth < 6.0))
        if len(xx) > args.max_points_per_frame:
            keep = rng.choice(len(xx), args.max_points_per_frame, replace=False)
            xx, yy = xx[keep], yy[keep]
        z = depth[yy, xx]
        cam = np.stack(((xx - cx) * z / fx, (yy - cy) * z / fy, z), axis=1)
        world = cam @ row["c2w"][:3, :3].T + row["c2w"][:3, 3]
        colors = rgb[yy, xx]
        for xyz, color in zip(world, colors):
            point_lines.append(f"{pid} {xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g} {int(color[0])} {int(color[1])} {int(color[2])} 0.0")
            pid += 1
    (out / "sparse/0/points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")
    print({"matched": len(associate(root)), "frames": len(rows), "train_frames": len(rows) - len(rows[::args.test_every]), "points": pid - 1, "output": str(out)})


if __name__ == "__main__":
    main()
