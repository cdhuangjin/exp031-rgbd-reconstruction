"""Auditable RGB 3D Gaussian baseline using the public gsplat rasterizer.

This is intentionally independent of Nerfstudio's FullImageDataManager. It
uses the standard Blender train/test split, known camera poses, RGB-only loss,
and reports held-out RGB PSNR/SSIM. No test images or depth maps are used for
optimization.
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
from PIL import Image

from gsplat_baseline import blender_intrinsics, blender_pose_to_viewmat, decode_blender_depth, select_frames


def _read_frames(root: Path, split: str) -> tuple[dict, list[dict]]:
    meta = json.loads((root / f"transforms_{split}.json").read_text(encoding="utf-8"))
    return meta, meta["frames"]


def _image_path(root: Path, frame: dict) -> Path:
    raw = Path(frame["file_path"])
    if raw.suffix:
        return root / raw
    return root / f"{raw}.png"


def _depth_path(root: Path, frame: dict) -> Path:
    stem = Path(frame["file_path"]).name
    candidates = sorted((root / "test").glob(f"{stem}_depth_*.png"))
    if not candidates:
        raise FileNotFoundError(f"no Blender depth image for {stem}")
    return candidates[0]


def _load_image(path: Path, resolution: int, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((resolution, resolution), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).to(device)


def _make_camera(meta: dict, frame: dict, resolution: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = int(meta.get("w", 800))
    height = int(meta.get("h", 800))
    fx, fy, cx, cy = blender_intrinsics(width, height, float(meta["camera_angle_x"]))
    scale_x = resolution / width
    scale_y = resolution / height
    K = torch.tensor(
        [[fx * scale_x, 0.0, cx * scale_x], [0.0, fy * scale_y, cy * scale_y], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )[None]
    viewmat = torch.from_numpy(blender_pose_to_viewmat(frame["transform_matrix"])).to(device)[None]
    return viewmat, K


def _ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    """Simple 3x3-window SSIM, sufficient for deterministic reporting."""
    import torch.nn.functional as F

    x = x.permute(2, 0, 1)[None]
    y = y.permute(2, 0, 1)[None]
    kernel = torch.ones((3, 1, 3, 3), device=x.device) / 9.0
    mu_x = F.conv2d(x, kernel, padding=1, groups=3)
    mu_y = F.conv2d(y, kernel, padding=1, groups=3)
    sigma_x = F.conv2d(x * x, kernel, padding=1, groups=3) - mu_x * mu_x
    sigma_y = F.conv2d(y * y, kernel, padding=1, groups=3) - mu_y * mu_y
    sigma_xy = F.conv2d(x * y, kernel, padding=1, groups=3) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    value = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
    return float(value.mean().detach().cpu())


def _render(means, quats, scales, opacities, colors, viewmat, K, resolution, render_mode="RGB"):
    from gsplat.rendering import rasterization

    render, alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=torch.exp(scales),
        opacities=torch.sigmoid(opacities).squeeze(-1),
        colors=torch.sigmoid(colors),
        viewmats=viewmat,
        Ks=K,
        width=resolution,
        height=resolution,
        packed=False,
        near_plane=0.01,
        far_plane=100.0,
        render_mode=render_mode,
        camera_model="pinhole",
    )
    if render_mode == "ED":
        return render[0, ..., 0], alpha[0]
    return render[0, ..., :3], alpha[0]


def run(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data).resolve()
    train_meta, train_frames_all = _read_frames(root, "train")
    _, test_frames_all = _read_frames(root, "test")
    train_frames = select_frames(train_frames_all, args.train_frames, args.seed)
    test_frames = select_frames(test_frames_all, args.eval_frames, args.seed + 1)

    means = torch.nn.Parameter((torch.rand(args.gaussians, 3, device=device) * 2.4) - 1.2)
    quats = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(args.gaussians, 1))
    scales = torch.nn.Parameter(torch.full((args.gaussians, 3), math.log(args.init_scale), device=device))
    opacities = torch.nn.Parameter(torch.full((args.gaussians, 1), -2.0, device=device))
    colors = torch.nn.Parameter(torch.zeros((args.gaussians, 3), device=device))
    optimizer = torch.optim.Adam([means, quats, scales, opacities, colors], lr=args.lr)

    start = time.time()
    losses = []
    for step in range(args.steps):
        frame = train_frames[step % len(train_frames)]
        viewmat, K = _make_camera(train_meta, frame, args.resolution, device)
        target = _load_image(_image_path(root, frame), args.resolution, device)
        pred, _ = _render(means, quats, scales, opacities, colors, viewmat, K, args.resolution)
        loss = (pred - target).abs().mean() + 0.1 * ((pred - target) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % args.log_every == 0:
            print(json.dumps({"step": step + 1, "loss": losses[-1]}, ensure_ascii=False), flush=True)

    psnrs, ssims, depth_rmses, depth_coverages = [], [], [], []
    with torch.no_grad():
        for frame in test_frames:
            viewmat, K = _make_camera(train_meta, frame, args.resolution, device)
            target = _load_image(_image_path(root, frame), args.resolution, device)
            pred, _ = _render(means, quats, scales, opacities, colors, viewmat, K, args.resolution)
            mse = float(((pred - target) ** 2).mean().detach().cpu())
            psnrs.append(-10.0 * math.log10(max(mse, 1e-12)))
            ssims.append(_ssim(pred.clamp(0, 1), target))
            depth_render, _ = _render(means, quats, scales, opacities, colors, viewmat, K, args.resolution, render_mode="ED")
            encoded = np.asarray(Image.open(_depth_path(root, frame)).convert("L"), dtype=np.uint8)
            depth_target = torch.from_numpy(decode_blender_depth(encoded, scale=4.0)).to(device)
            depth_target = torch.nn.functional.interpolate(depth_target[None, None], size=(args.resolution, args.resolution), mode="nearest")[0, 0]
            valid = (depth_target > 0.02) & (depth_target < 5.0) & torch.isfinite(depth_render)
            coverage = float(valid.float().mean().detach().cpu())
            if bool(valid.any()):
                depth_rmses.append(float(torch.sqrt(((depth_render[valid] - depth_target[valid]) ** 2).mean()).detach().cpu()))
                depth_coverages.append(coverage)

    result = {
        "status": "PASS",
        "method": "gsplat_rgb_optimization_baseline",
        "dataset": root.name,
        "seed": args.seed,
        "train_frames": len(train_frames),
        "eval_frames": len(test_frames),
        "resolution": args.resolution,
        "gaussians": args.gaussians,
        "steps": args.steps,
        "device": str(device),
        "mean_train_loss_last10": float(np.mean(losses[-10:])),
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_std": float(np.std(psnrs, ddof=1)) if len(psnrs) > 1 else 0.0,
        "ssim_mean": float(np.mean(ssims)),
        "ssim_std": float(np.std(ssims, ddof=1)) if len(ssims) > 1 else 0.0,
        "depth_rmse_mean_m": float(np.mean(depth_rmses)) if depth_rmses else float("nan"),
        "depth_rmse_std_m": float(np.std(depth_rmses, ddof=1)) if len(depth_rmses) > 1 else 0.0,
        "depth_coverage_mean": float(np.mean(depth_coverages)) if depth_coverages else 0.0,
        "runtime_sec": time.time() - start,
        "test_images_used_for_optimization": False,
        "test_depth_used_for_optimization": False,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--train-frames", type=int, default=50)
    parser.add_argument("--eval-frames", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--gaussians", type=int, default=2000)
    parser.add_argument("--init-scale", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--log-every", type=int, default=50)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
