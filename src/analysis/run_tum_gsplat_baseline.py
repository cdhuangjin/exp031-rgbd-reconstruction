"""Same-protocol TUM RGB-D Gaussian reference using the public gsplat rasterizer.

The model uses ground-truth TUM poses and training RGB-D frames only to
initialize a colored Gaussian cloud. Optimization is RGB-only; held-out RGB
and depth are never used for optimization. This is a reference baseline, not
a claim of a full SOTA 3DGS implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


INTRINSICS = {
    "freiburg1": (517.3, 516.5, 318.6, 255.3),
    "freiburg2": (520.9, 521.0, 325.1, 249.7),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
}


def _lines(path: Path) -> list[tuple[float, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            rows.append((float(parts[0]), parts[1]))
    return rows


def _nearest(ts: float, rows: list[tuple[float, str]], max_delta: float) -> str | None:
    if not rows:
        return None
    pos = min(range(len(rows)), key=lambda i: abs(rows[i][0] - ts))
    return rows[pos][1] if abs(rows[pos][0] - ts) <= max_delta else None


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float32)


def _associate(root: Path) -> list[dict]:
    rgb = _lines(root / "rgb.txt")
    depth = _lines(root / "depth.txt")
    gt = []
    for line in (root / "groundtruth.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) == 8:
            gt.append((float(p[0]), np.asarray([float(x) for x in p[1:]], dtype=np.float32)))
    result = []
    for ts, rgb_rel in rgb:
        depth_rel = _nearest(ts, depth, 0.025)
        if depth_rel is None or not gt:
            continue
        gi = min(range(len(gt)), key=lambda i: abs(gt[i][0] - ts))
        if abs(gt[gi][0] - ts) > 0.05:
            continue
        v = gt[gi][1]
        tx, ty, tz, qx, qy, qz, qw = [float(x) for x in v]
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = _quat_to_rot(qx, qy, qz, qw)
        c2w[:3, 3] = [tx, ty, tz]
        result.append({"ts": ts, "rgb": rgb_rel, "depth": depth_rel, "c2w": c2w})
    return result


def _pick(rows: list[dict], count: int, offset: int) -> list[dict]:
    if not rows:
        return []
    if count >= len(rows):
        return rows
    idx = np.linspace(offset, len(rows) - 1, count).round().astype(int)
    return [rows[int(i)] for i in idx]


def _camera(root: Path, c2w: np.ndarray, width: int, height: int, device: torch.device):
    name = root.name
    family = "freiburg3" if "freiburg3" in name else ("freiburg2" if "freiburg2" in name else "freiburg1")
    fx, fy, cx, cy = INTRINSICS[family]
    sx, sy = width / 640.0, height / 480.0
    K = torch.tensor([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], dtype=torch.float32, device=device)[None]
    view = np.linalg.inv(c2w).astype(np.float32)
    return torch.from_numpy(view).to(device)[None], K


def _load(path: Path, width: int, height: int, device: torch.device, depth: bool = False):
    image = Image.open(path)
    if depth:
        arr = np.asarray(image, dtype=np.float32) / 5000.0
        out = Image.fromarray(arr).resize((width, height), Image.Resampling.NEAREST)
        return torch.from_numpy(np.asarray(out, dtype=np.float32).copy()).to(device)
    image = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).to(device)


def _ssim(x, y):
    x, y = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]
    k = torch.ones((3, 1, 3, 3), device=x.device) / 9.0
    mx, my = F.conv2d(x, k, padding=1, groups=3), F.conv2d(y, k, padding=1, groups=3)
    vx = F.conv2d(x * x, k, padding=1, groups=3) - mx * mx
    vy = F.conv2d(y * y, k, padding=1, groups=3) - my * my
    c = F.conv2d(x * y, k, padding=1, groups=3) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float((((2 * mx * my + c1) * (2 * c + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))).mean().detach().cpu())


def _render(means, quats, scales, opacities, colors, view, K, width, height, mode="RGB"):
    from gsplat.rendering import rasterization
    render, alpha, _ = rasterization(means=means, quats=quats, scales=torch.exp(scales),
        opacities=torch.sigmoid(opacities).squeeze(-1), colors=torch.sigmoid(colors),
        viewmats=view, Ks=K, width=width, height=height, packed=False,
        near_plane=0.05, far_plane=8.0, render_mode=mode, camera_model="pinhole")
    return (render[0, ..., 0] if mode == "ED" else render[0, ..., :3]), alpha[0]


def run(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    root, device = Path(args.data).resolve(), torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = _associate(root)
    train, test = _pick(rows, args.train_frames, 0), _pick(rows, args.eval_frames, 1)
    train_ts = {r["ts"] for r in train}; test = [r for r in test if r["ts"] not in train_ts]
    width, height = args.width, args.height
    # Colored RGB-D initialization from training frames; optimization remains RGB-only.
    xyz, rgb = [], []
    for row in train[::max(1, len(train) // 12)]:
        d = _load(root / row["depth"], width, height, device, depth=True)
        im = _load(root / row["rgb"], width, height, device)
        fx, fy, cx, cy = INTRINSICS["freiburg3" if "freiburg3" in root.name else ("freiburg2" if "freiburg2" in root.name else "freiburg1")]
        ys, xs = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        z = d[::args.pixel_stride, ::args.pixel_stride]; xx = xs[::args.pixel_stride, ::args.pixel_stride]; yy = ys[::args.pixel_stride, ::args.pixel_stride]
        valid = (z > 0.2) & (z < 6.0)
        cam = torch.stack(((xx - cx * width / 640) * z / (fx * width / 640), (yy - cy * height / 480) * z / (fy * height / 480), z), -1)[valid]
        c2w = torch.from_numpy(row["c2w"]).to(device); world = cam @ c2w[:3, :3].T + c2w[:3, 3]
        xyz.append(world); rgb.append(im[::args.pixel_stride, ::args.pixel_stride][valid])
    xyz, rgb = torch.cat(xyz), torch.cat(rgb)
    if len(xyz) > args.gaussians:
        keep = torch.randperm(len(xyz), device=device)[:args.gaussians]; xyz, rgb = xyz[keep], rgb[keep]
    elif len(xyz) < args.gaussians:
        keep = torch.randint(len(xyz), (args.gaussians - len(xyz),), device=device); xyz, rgb = torch.cat([xyz, xyz[keep]], 0), torch.cat([rgb, rgb[keep]], 0)
    means = torch.nn.Parameter(xyz); quats = torch.nn.Parameter(torch.tensor([1., 0, 0, 0], device=device).repeat(len(xyz), 1))
    scales = torch.nn.Parameter(torch.full((len(xyz), 3), math.log(args.init_scale), device=device)); opacities = torch.nn.Parameter(torch.full((len(xyz), 1), -1.0, device=device)); colors = torch.nn.Parameter(torch.logit(rgb.clamp(.02, .98)))
    opt = torch.optim.Adam([means, quats, scales, opacities, colors], lr=args.lr); start = time.time(); losses = []
    for step in range(args.steps):
        row = train[step % len(train)]; view, K = _camera(root, row["c2w"], width, height, device); target = _load(root / row["rgb"], width, height, device)
        pred, _ = _render(means, quats, scales, opacities, colors, view, K, width, height); loss = (pred - target).abs().mean() + .1 * ((pred - target) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
    psnr, ssim, drmse, cov = [], [], [], []
    with torch.no_grad():
        for row in test:
            view, K = _camera(root, row["c2w"], width, height, device); target = _load(root / row["rgb"], width, height, device); pred, _ = _render(means, quats, scales, opacities, colors, view, K, width, height)
            mse = float(((pred - target) ** 2).mean().cpu()); psnr.append(-10 * math.log10(max(mse, 1e-12))); ssim.append(_ssim(pred.clamp(0, 1), target))
            dpred, _ = _render(means, quats, scales, opacities, colors, view, K, width, height, "ED"); dgt = _load(root / row["depth"], width, height, device, depth=True); valid = (dgt > .2) & (dgt < 6.0) & torch.isfinite(dpred); cov.append(float(valid.float().mean().cpu()))
            if valid.any(): drmse.append(float(torch.sqrt(((dpred[valid] - dgt[valid]) ** 2).mean()).cpu()))
    out = {"status":"PASS", "method":"tum_gsplat_rgb_only_with_train_rgbd_init", "dataset":root.name, "seed":args.seed, "matched_frames":len(rows), "train_frames":len(train), "eval_frames":len(test), "width":width, "height":height, "gaussians":len(xyz), "steps":args.steps, "device":str(device), "psnr_mean_db":float(np.mean(psnr)), "psnr_std_db":float(np.std(psnr, ddof=1)) if len(psnr)>1 else 0., "ssim_mean":float(np.mean(ssim)), "ssim_std":float(np.std(ssim, ddof=1)) if len(ssim)>1 else 0., "depth_rmse_mean_m":float(np.mean(drmse)) if drmse else float("nan"), "depth_rmse_std_m":float(np.std(drmse, ddof=1)) if len(drmse)>1 else 0., "depth_coverage_mean":float(np.mean(cov)), "runtime_sec":time.time()-start, "test_rgb_and_depth_used_for_optimization":False, "train_depth_used_for_initialization":True}
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8"); print(json.dumps(out, indent=2), flush=True); return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--output", required=True); p.add_argument("--steps", type=int, default=1000); p.add_argument("--train-frames", type=int, default=50); p.add_argument("--eval-frames", type=int, default=20); p.add_argument("--width", type=int, default=160); p.add_argument("--height", type=int, default=120); p.add_argument("--gaussians", type=int, default=2000); p.add_argument("--pixel-stride", type=int, default=3); p.add_argument("--init-scale", type=float, default=.03); p.add_argument("--lr", type=float, default=.01); p.add_argument("--seed", type=int, default=31); run(p.parse_args())
