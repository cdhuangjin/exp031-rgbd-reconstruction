import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))

from local_smoke import build_local_report


class LocalSmokeTests(unittest.TestCase):
    def test_missing_optional_dataset_is_reported_as_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            report = build_local_report(tum_root=Path(directory) / "missing")
            self.assertEqual(report["tum"], [])
            self.assertEqual(len(report["warnings"]), 1)


if __name__ == "__main__":
    unittest.main()
