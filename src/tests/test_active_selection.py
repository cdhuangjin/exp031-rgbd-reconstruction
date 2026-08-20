import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import unittest

from active_selection import CandidateView, perturb_pose_errors, select_next_view


class ActiveSelectionTests(unittest.TestCase):
  def test_pose_perturbations_are_deterministic_and_have_expected_shape(self):
    first = perturb_pose_errors(4, translation_std=0.02, rotation_std_deg=3.0, seed=7)
    second = perturb_pose_errors(4, translation_std=0.02, rotation_std_deg=3.0, seed=7)

    self.assertEqual(first, second)
    self.assertEqual(len(first), 4)
    self.assertTrue(all(len(error) == 6 for error in first))


  def test_full_risk_selection_prefers_stable_view_over_nominally_better_fragile_view(self):
    candidates = [
        CandidateView("fragile", nominal_gain=1.00, pose_gain_std=0.40, uncovered_risk=0.10, cost=0.10),
        CandidateView("stable", nominal_gain=0.90, pose_gain_std=0.05, uncovered_risk=0.05, cost=0.10),
    ]

    selected = select_next_view(candidates, mode="full_risk", lambda_pose=1.0)

    self.assertEqual(selected.view_id, "stable")


  def test_invalid_mode_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "mode"):
        select_next_view(
            [CandidateView("v0", nominal_gain=0.1, pose_gain_std=0.1, uncovered_risk=0.1, cost=0.1)],
            mode="unknown",
        )
