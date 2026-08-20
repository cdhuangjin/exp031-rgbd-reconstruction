from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from e0_smoke import run


def _gpu_metadata() -> dict[str, object]:
    metadata: dict[str, object] = {"python": sys.version, "platform": platform.platform()}
    try:
        import torch

        metadata.update(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
            }
        )
        if torch.cuda.is_available():
            metadata.update(
                {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
                }
            )
    except Exception as exc:  # metadata must never prevent a smoke run
        metadata["torch_probe_error"] = repr(exc)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=12)
    args = parser.parse_args()

    config = {
        "seed": args.seed,
        "candidate_count": args.candidate_count,
        "max_steps": args.max_steps,
        "mode": "full_risk",
        "alpha": 0.1,
        "lambda_pose": 1.0,
        "lambda_cost": 0.1,
        "lambda_cov": 0.5,
        "initial_quality": 0.35,
        "target_quality": 0.70,
        "max_risk": 0.20,
    }
    start = time.perf_counter()
    result = run(config)
    result["runner_wall_time_seconds"] = time.perf_counter() - start
    payload = {"config": config, "environment": _gpu_metadata(), "result": result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
