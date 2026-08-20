from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_manifest import inspect_blender_scene, inspect_tum_archive


def build_local_report(
    *,
    tum_root: str | Path | None = None,
    blender_root: str | Path | None = None,
) -> dict[str, object]:
    report: dict[str, object] = {"tum": [], "blender": [], "warnings": []}

    if tum_root is not None:
        tum_path = Path(tum_root)
        if not tum_path.is_dir():
            report["warnings"].append(f"TUM directory not found: {tum_path}")
        else:
            for archive in sorted(tum_path.glob("*.tgz")):
                report["tum"].append(inspect_tum_archive(archive))

    if blender_root is not None:
        blender_path = Path(blender_root)
        if not blender_path.is_dir():
            report["warnings"].append(f"Blender directory not found: {blender_path}")
        else:
            for scene in sorted(path for path in blender_path.iterdir() if path.is_dir()):
                try:
                    report["blender"].append(inspect_blender_scene(scene))
                except (FileNotFoundError, ValueError) as exc:
                    report["warnings"].append(str(exc))

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tum-root", type=Path)
    parser.add_argument("--blender-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_local_report(tum_root=args.tum_root, blender_root=args.blender_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
