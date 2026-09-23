"""Invalid report requests should fail clearly without an internal traceback."""

import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class ReportCliErrorsTest(unittest.TestCase):
    def test_unknown_gid_returns_error_without_report_or_traceback(self):
        known = set(pd.read_parquet(ROOT / "data" / "nodes.parquet").gid)
        missing = 0
        while missing in known:
            missing += 1
        result = subprocess.run(
            [sys.executable, "-X", "utf8", str(ROOT / "report_node.py"),
             str(missing), "--data", str(ROOT / "data")],
            capture_output=True, text=True, encoding="utf-8", timeout=120,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn(f"gid {missing} отсутствует в выгрузке", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
