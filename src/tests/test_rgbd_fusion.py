import unittest

import numpy as np

from rgbd_fusion import (
    depth_to_camera_points,
    depth_error_metrics,
    select_frame_indices,
    voxel_fuse,
    voxel_fuse_torch,
)


class RGBDFusionTests(unittest.TestCase):
    def test_select_frame_indices_is_deterministic_and_includes_endpoints(self):
        self.assertEqual(select_frame_indices(10, stride=3, max_frames=4), [0, 3, 6, 9])
        self.assertEqual(select_frame_indices(3, stride=10, max_frames=10), [0, 1, 2])

    def test_depth_to_camera_points_filters_invalid_depth(self):
        depth = np.array([[0.0, 1.0], [2.0, 5.0]], dtype=np.float32)
        points, pixels = depth_to_camera_points(
            depth, fx=2.0, fy=2.0, cx=0.0, cy=0.0, depth_min=0.5, depth_max=3.0
        )
        self.assertEqual(points.shape, (2, 3))
        np.testing.assert_allclose(points[:, 2], [1.0, 2.0])
        np.testing.assert_array_equal(pixels, [[0, 1], [1, 0]])

    def test_voxel_fuse_returns_weighted_centroid_and_coverage(self):
        points = np.array([[0.01, 0.0, 1.0], [0.03, 0.0, 1.0], [0.20, 0.0, 1.0]])
        weights = np.array([1.0, 3.0, 1.0])
        fused = voxel_fuse(points, weights, voxel_size=0.1)
        self.assertEqual(fused["voxel_count"], 2)
        self.assertAlmostEqual(fused["weighted_centroid"][0][0], 0.025)
        self.assertAlmostEqual(fused["coverage"], 2 / 3)

    def test_torch_voxel_fuse_matches_numpy_voxel_count_on_cpu(self):
        points = np.array([[0.01, 0.0, 1.0], [0.03, 0.0, 1.0], [0.20, 0.0, 1.0]])
        weights = np.ones(3)
        fused = voxel_fuse_torch(points, weights, voxel_size=0.1, device="cpu")
        self.assertEqual(fused["voxel_count"], 2)
        self.assertAlmostEqual(fused["coverage"], 2 / 3)
        self.assertAlmostEqual(fused["weighted_centroid"][0][0], 0.02)

    def test_depth_error_metrics_uses_only_jointly_valid_pixels(self):
        predicted = np.array([[1.0, 2.0], [0.0, 4.0]])
        ground_truth = np.array([[1.5, 2.0], [3.0, 0.0]])
        metrics = depth_error_metrics(predicted, ground_truth, depth_min=0.5, depth_max=5.0)
        self.assertAlmostEqual(metrics["rmse"], (0.5**2 / 2) ** 0.5)
        self.assertAlmostEqual(metrics["mae"], 0.25)
        self.assertAlmostEqual(metrics["coverage"], 2 / 4)

    def test_projection_uses_eval_pose_when_present(self):
        from run_rgbd_fusion import _project_world_points

        points = np.asarray([[0.5, 0.0, 1.0]], dtype=np.float64)
        true_pose = np.eye(4)
        perturbed_pose = np.eye(4)
        perturbed_pose[0, 3] = 1.0
        record = {
            "depth": np.ones((5, 5), dtype=np.float32),
            "pose": perturbed_pose,
            "eval_pose": true_pose,
            "intrinsics": (2.0, 2.0, 2.0, 2.0),
        }

        projected = _project_world_points(points, record, depth_min=0.3, depth_max=5.0)

        self.assertTrue(np.isfinite(projected[2, 3]))
        self.assertFalse(np.isfinite(projected[2, 1]))


if __name__ == "__main__":
    unittest.main()
