"""Vanilla 3DGS-style RGB reference using the installed public gsplat API.

This is deliberately separate from the lightweight fusion method and from
the simplified RGB-only reference. It uses SH degree 3 and the public
DefaultStrategy densification/pruning callbacks. Training depth is used only
to initialize colored points; held-out RGB/depth are never optimized.
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

from run_tum_gsplat_baseline import INTRINSICS, _associate, _camera, _load, _pick, _ssim


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / 0.28209479177387814


def render(params, view, K, width, height, mode="RGB"):
    from gsplat.rendering import rasterization

    colors, sh_degree = params["shs"], 3
    renders, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]).squeeze(-1),
        colors=colors,
        viewmats=view,
        Ks=K,
        width=width,
        height=height,
        near_plane=0.05,
        far_plane=8.0,
        render_mode=mode,
        packed=False,
        sh_degree=sh_degree,
        camera_model="pinhole",
    )
    return (renders[0, ..., 0] if mode == "ED" else renders[0, ..., :3]), alpha[0], info


def initial_cloud(root, train, width, height, device, gaussians, pixel_stride):
    xyz, rgb = [], []
    family = next((k for k in INTRINSICS if k in root.name), "freiburg1")
    fx, fy, cx, cy = INTRINSICS[family]
    ys, xs = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    for row in train[::max(1, len(train) // 12)]:
        depth = _load(root / row["depth"], width, height, device, depth=True)
        image = _load(root / row["rgb"], width, height, device)
        z = depth[::pixel_stride, ::pixel_stride]
        xx = xs[::pixel_stride, ::pixel_stride]
        yy = ys[::pixel_stride, ::pixel_stride]
        valid = (z > 0.2) & (z < 6.0)
        cam = torch.stack(((xx - cx * width / 640) * z / (fx * width / 640), (yy - cy * height / 480) * z / (fy * height / 480), z), -1)[valid]
        c2w = torch.from_numpy(row["c2w"]).to(device)
        xyz.append(cam @ c2w[:3, :3].T + c2w[:3, 3])
        rgb.append(image[::pixel_stride, ::pixel_stride][valid])
    xyz, rgb = torch.cat(xyz), torch.cat(rgb)
    if len(xyz) > gaussians:
        keep = torch.randperm(len(xyz), device=device)[:gaussians]
        xyz, rgb = xyz[keep], rgb[keep]
    elif len(xyz) < gaussians:
        keep = torch.randint(len(xyz), (gaussians - len(xyz),), device=device)
        xyz, rgb = torch.cat([xyz, xyz[keep]]), torch.cat([rgb, rgb[keep]])
    n = len(xyz)
    return {
        "means": torch.nn.Parameter(xyz),
        "quats": torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1)),
        "scales": torch.nn.Parameter(torch.full((n, 3), math.log(0.03), device=device)),
        "opacities": torch.nn.Parameter(torch.full((n, 1), -1.0, device=device)),
        "shs": torch.nn.Parameter(torch.cat([rgb_to_sh(rgb)[:, None, :], torch.zeros((n, 15, 3), device=device)], dim=1)),
    }


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.data).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = _associate(root)
    train, test = _pick(rows, args.train_frames, 0), _pick(rows, args.eval_frames, 1)
    train_ts = {r["ts"] for r in train}
    test = [r for r in test if r["ts"] not in train_ts]
    width, height = args.width, args.height
    params = initial_cloud(root, train, width, height, device, args.gaussians, args.pixel_stride)
    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=1.6e-4),
        "quats": torch.optim.Adam([params["quats"]], lr=1e-3),
        "scales": torch.optim.Adam([params["scales"]], lr=5e-3),
        "opacities": torch.optim.Adam([params["opacities"]], lr=5e-2),
        "shs": torch.optim.Adam([params["shs"]], lr=2.5e-3),
    }
    from gsplat.strategy import DefaultStrategy

    strategy = DefaultStrategy(refine_start_iter=max(50, args.steps // 5), refine_stop_iter=max(100, args.steps - 50), refine_every=max(25, args.steps // 10), reset_every=max(100, args.steps // 3), verbose=False)
    state = strategy.initialize_state(scene_scale=1.0)
    start = time.time()
    for step in range(args.steps):
        row = train[step % len(train)]
        view, K = _camera(root, row["c2w"], width, height, device)
        target = _load(root / row["rgb"], width, height, device)
        pred, _, info = render(params, view, K, width, height)
        loss = (pred - target).abs().mean() + 0.1 * ((pred - target) ** 2).mean()
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
    psnr, ssim, drmse, cov = [], [], [], []
    with torch.no_grad():
        for row in test:
            view, K = _camera(root, row["c2w"], width, height, device)
            target = _load(root / row["rgb"], width, height, device)
            pred, _, _ = render(params, view, K, width, height)
            mse = float(((pred - target) ** 2).mean().cpu())
            psnr.append(-10 * math.log10(max(mse, 1e-12)))
            ssim.append(_ssim(pred.clamp(0, 1), target))
            dpred, _, _ = render(params, view, K, width, height, "ED")
            dgt = _load(root / row["depth"], width, height, device, depth=True)
            valid = (dgt > 0.2) & (dgt < 6.0) & torch.isfinite(dpred)
            cov.append(float(valid.float().mean().cpu()))
            if valid.any():
                drmse.append(float(torch.sqrt(((dpred[valid] - dgt[valid]) ** 2).mean()).cpu()))
    out = {
        "status": "PASS",
        "method": "tum_gsplat_vanilla_sh3_default_strategy",
        "dataset": root.name,
        "seed": args.seed,
        "matched_frames": len(rows),
        "train_frames": len(train),
        "eval_frames": len(test),
        "width": width,
        "height": height,
        "initial_gaussians": args.gaussians,
        "final_gaussians": len(params["means"]),
        "steps": args.steps,
        "sh_degree": 3,
        "densification": "DefaultStrategy",
        "device": str(device),
        "psnr_mean_db": float(np.mean(psnr)) if psnr else float("nan"),
        "ssim_mean": float(np.mean(ssim)) if ssim else float("nan"),
        "depth_rmse_mean_m": float(np.mean(drmse)) if drmse else float("nan"),
        "depth_coverage_mean": float(np.mean(cov)) if cov else float("nan"),
        "runtime_sec": time.time() - start,
        "test_rgb_and_depth_used_for_optimization": False,
        "train_depth_used_for_initialization": True,
    }
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--train-frames", type=int, default=50)
    p.add_argument("--eval-frames", type=int, default=20)
    p.add_argument("--width", type=int, default=160)
    p.add_argument("--height", type=int, default=120)
    p.add_argument("--gaussians", type=int, default=2000)
    p.add_argument("--pixel-stride", type=int, default=3)
    p.add_argument("--seed", type=int, default=31)
    run(p.parse_args())
