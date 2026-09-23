import unittest
from pathlib import Path

from check_viewer import check_viewer, extract_payload


ROOT = Path(__file__).resolve().parents[1]


class ViewerSelfCheckTest(unittest.TestCase):
    def test_check_generated_viewer_payload(self):
        result = check_viewer(ROOT / "out" / "network.html")
        self.assertGreater(result["nodes"], 0)
        self.assertGreater(result["edges"], 0)
        self.assertGreater(result["clusters"], 0)

    def test_placeholder_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "placeholder"):
            extract_payload("const data = /* GRAPH_DATA */ null;")


if __name__ == "__main__":
    unittest.main()
