"""The report CLI must honor --data independently of its working directory."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from graph_tools import GraphIndex
from report_node import node_report


ROOT = Path(__file__).resolve().parents[1]


class ReportCliPathsTest(unittest.TestCase):
    def test_explicit_data_path_works_outside_repository(self):
        index = GraphIndex(ROOT / "data")
        gid = index.top_nodes(limit=1)["shown"][0]["gid"]
        expected = node_report(index, gid)
        with tempfile.TemporaryDirectory() as working_dir:
            result = subprocess.run(
                [sys.executable, "-X", "utf8", str(ROOT / "report_node.py"),
                 gid, "--data", str(ROOT / "data")],
                cwd=working_dir, capture_output=True, text=True,
                encoding="utf-8", timeout=120,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.rstrip("\n"), expected)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
