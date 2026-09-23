import tempfile
import unittest
from pathlib import Path

import pandas as pd

from compare_outputs import compare_outputs


class CompareOutputsTest(unittest.TestCase):
    def write_nodes(self, path: Path, rows):
        path.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(path / "nodes_roles.csv", index=False)

    def test_compare_outputs_marks_added_removed_and_changed_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = [
                {"gid": "1", "role": "peripheral", "cluster_id": 1, "priority_score": .10,
                 "evidence": "old one", "attention": "нет"},
                {"gid": "2", "role": "terminal", "cluster_id": 1, "priority_score": .20,
                 "evidence": "old two", "attention": "нет"},
            ]
            updated = [
                {"gid": "1", "role": "consolidator", "cluster_id": 2, "priority_score": .72,
                 "evidence": "new one", "attention": "flag"},
                {"gid": "3", "role": "terminal", "cluster_id": 3, "priority_score": .40,
                 "evidence": "new three", "attention": "нет"},
            ]
            self.write_nodes(root / "old", base)
            self.write_nodes(root / "new", updated)
            changes = compare_outputs(root / "old", root / "new").set_index("gid")

        self.assertEqual(set(changes.index), {"1", "2", "3"})
        self.assertEqual(changes.loc["1", "status"], "kept")
        self.assertTrue(changes.loc["1", "role_changed"])
        self.assertTrue(changes.loc["1", "cluster_changed"])
        self.assertTrue(changes.loc["1", "attention_changed"])
        self.assertAlmostEqual(changes.loc["1", "priority_delta"], .62)
        self.assertEqual(changes.loc["2", "status"], "removed")
        self.assertEqual(changes.loc["3", "status"], "new")


if __name__ == "__main__":
    unittest.main()
