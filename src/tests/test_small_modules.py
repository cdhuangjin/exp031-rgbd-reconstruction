import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from small_modules import (
    combine_frame_reliability,
    depth_validity_weights,
    pose_frame_weight,
    scene_adaptive_threshold,
)


class SmallModuleTests(unittest.TestCase):
    def test_depth_gate_rejects_invalid_and_out_of_range_depth(self):
        weights = depth_validity_weights(
            [0.0, 0.5, 2.0, 5.0], min_depth=0.2, max_depth=3.0
        )
        self.assertEqual(weights, [0.0, 1.0, 1.0, 0.0])

    def test_pose_weight_decreases_with_motion_and_reprojection_error(self):
        stable = pose_frame_weight(0.01, 1.0, 0.01)
        unstable = pose_frame_weight(0.20, 10.0, 0.50)
        self.assertGreater(stable, unstable)
        self.assertLessEqual(stable, 1.0)
        self.assertGreaterEqual(unstable, 0.0)

    def test_scene_threshold_uses_source_validation_errors(self):
        self.assertAlmostEqual(scene_adaptive_threshold([0.1, 0.2, 0.3, 0.4], quantile=0.75), 0.3)

    def test_frame_reliability_is_bounded_product(self):
        self.assertAlmostEqual(combine_frame_reliability(0.8, 0.5), 0.4)


if __name__ == "__main__":
    unittest.main()
