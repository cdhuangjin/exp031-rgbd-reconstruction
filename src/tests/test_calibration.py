import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import unittest

from calibration import calibration_radius, should_stop


class CalibrationTests(unittest.TestCase):
  def test_calibration_radius_is_nonnegative_quantile_of_absolute_residuals(self):
    self.assertGreaterEqual(calibration_radius([0.1, -0.2, 0.3], alpha=0.1), 0.3)


  def test_stop_requires_quality_lower_bound_and_low_risk(self):
    radius = calibration_radius([0.05, 0.10, 0.15], alpha=0.1)

    self.assertTrue(should_stop(predicted_quality=0.95, radius=radius, target_quality=0.70, uncovered_risk=0.10, max_risk=0.20))
    self.assertFalse(should_stop(predicted_quality=0.72, radius=radius, target_quality=0.70, uncovered_risk=0.10, max_risk=0.20))
    self.assertFalse(should_stop(predicted_quality=0.95, radius=radius, target_quality=0.70, uncovered_risk=0.30, max_risk=0.20))


  def test_empty_residuals_are_rejected(self):
    with self.assertRaisesRegex(ValueError, "residual"):
        calibration_radius([], alpha=0.1)
