import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))

from dataset_manifest import (
    inspect_blender_scene,
    inspect_tum_archive,
    nearest_timestamp_pairs,
    sample_candidate_indices,
)


class DatasetManifestTests(unittest.TestCase):
    def test_blender_scene_requires_all_transform_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val", "test"):
                (root / f"transforms_{split}.json").write_text(
                    json.dumps({"frames": [{"file_path": f"{split}/r_000"}]}),
                    encoding="utf-8",
                )
            report = inspect_blender_scene(root)
            self.assertEqual(report["frame_counts"], {"train": 1, "val": 1, "test": 1})

    def test_tum_archive_reports_required_members_without_extracting(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "desk.tgz"
            with tarfile.open(archive, "w:gz") as handle:
                for name in (
                    "desk/rgb.txt",
                    "desk/depth.txt",
                    "desk/groundtruth.txt",
                    "desk/rgb/000.png",
                    "desk/depth/000.png",
                ):
                    info = tarfile.TarInfo(name)
                    info.size = 0
                    handle.addfile(info)
            report = inspect_tum_archive(archive)
            self.assertTrue(report["has_rgb_index"])
            self.assertTrue(report["has_depth_index"])
            self.assertTrue(report["has_groundtruth"])

    def test_timestamp_pairing_is_nearest_and_respects_tolerance(self):
        rgb = [(0.00, "rgb0"), (0.10, "rgb1")]
        depth = [(0.01, "depth0"), (0.30, "depth1")]
        self.assertEqual(nearest_timestamp_pairs(rgb, depth, tolerance=0.02), [("rgb0", "depth0")])

    def test_candidate_sampling_keeps_initial_frames_and_is_deterministic(self):
        self.assertEqual(sample_candidate_indices(10, initial_count=3, step=2), [0, 1, 2, 4, 6, 8])


if __name__ == "__main__":
    unittest.main()
