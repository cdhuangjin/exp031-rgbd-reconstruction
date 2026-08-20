import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from module_smoke import run_smoke


class ModuleSmokeTests(unittest.TestCase):
    def test_smoke_is_deterministic_and_writes_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "module_smoke.json"
            first = run_smoke(output=output, seed=7)
            second = run_smoke(output=None, seed=7)

            self.assertEqual(first, second)
            self.assertEqual(first["source_threshold"], 0.09)
            self.assertEqual(first["depth_weights"], [0.0, 1.0, 1.0, 0.0])
            self.assertEqual(len(first["pose_weights"]), 3)
            self.assertEqual(len(first["frame_reliability"]), 3)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in first["frame_reliability"]))
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), first)


if __name__ == "__main__":
    unittest.main()
