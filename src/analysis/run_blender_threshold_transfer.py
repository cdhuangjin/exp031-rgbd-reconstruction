"""Audit frozen versus target-calibrated reliability thresholds on Blender.

The target-calibrated threshold is estimated only from build frames (the
held-out every-fourth frames are excluded), so it tests domain adaptation
without leaking evaluation depths. It is an analysis of the threshold module,
not an additional headline method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_blender_depth_fusion import load_scene, run_prepared
from small_modules import scene_adaptive_threshold


def run(args: argparse.Namespace) -> dict:
    root = Path(args.root).resolve()
    scenes = sorted(x for x in root.iterdir() if x.is_dir() and (x / "transforms_test.json").exists())
    prepared = [load_scene(scene, max_frames=args.max_frames, stride=args.stride) for scene in scenes]
    source_errors = [r["invalid_ratio"] for item in prepared[:2] for r in item["records"]]
    frozen = scene_adaptive_threshold(source_errors, quantile=args.quantile)
    rows = []
    for item in prepared:
        build_errors = [r["invalid_ratio"] for i, r in enumerate(item["records"]) if i % 4 != 0]
        adaptive = scene_adaptive_threshold(build_errors, quantile=args.quantile)
        for label, threshold in (("frozen_source", frozen), ("target_build_calibrated", adaptive)):
            for variant in ("baseline", "full"):
                result = run_prepared(item, variant=variant, threshold=threshold)
                result["threshold_mode"] = label
                result["threshold_source"] = "source_scenes" if label == "frozen_source" else "target_build_frames"
                rows.append(result)
    payload = {
        "status": "PASS_WITH_DOMAIN_FAILURES" if any(r["heldout_frames"] == 0 for r in rows) else "PASS",
        "config": {"root": str(root), "max_frames": args.max_frames, "stride": args.stride, "quantile": args.quantile, "heldout_rule": "every fourth sampled frame", "heldout_depth_used_for_threshold": False},
        "source_scenes": [x["sequence"] for x in prepared[:2]],
        "frozen_source_threshold": frozen,
        "results": rows,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(rows), "output": str(args.output)}, indent=2))
    return payload


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=48)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--quantile", type=float, default=0.9)
    run(p.parse_args())
