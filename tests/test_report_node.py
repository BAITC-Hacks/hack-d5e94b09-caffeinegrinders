import unittest
from pathlib import Path

from graph_tools import GraphIndex
from report_node import node_report


ROOT = Path(__file__).resolve().parents[1]


class NodeReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = GraphIndex(ROOT / "data")

    def test_node_report_contains_profile_sections(self):
        gid = self.index.top_nodes(limit=1)["shown"][0]["gid"]
        text = node_report(self.index, gid)
        self.assertIn(f"gid {gid}", text)
        self.assertIn("role:", text)
        self.assertIn("evidence:", text)
        self.assertIn("direct counterparties:", text)


if __name__ == "__main__":
    unittest.main()
