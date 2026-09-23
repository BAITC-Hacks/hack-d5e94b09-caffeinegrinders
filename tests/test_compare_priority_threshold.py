"""Regression checks for compare priority threshold."""

import unittest
import pandas as pd
import tempfile
from pathlib import Path
from compare_outputs import compare_outputs


class RegressionTest(unittest.TestCase):

    def test_priority_changes_at_threshold_are_included_in_both_directions(self):
        old = pd.DataFrame({
            "gid": [1, 2, 3, 4, 5], "role": ["transit"] * 5,
            "cluster_id": [1] * 5, "priority_score": [.5] * 5,
            "evidence": ["reason"] * 5, "attention": ["нет"] * 5,
        })
        new = old.copy()
        new["priority_score"] = [.55, .45, .549, .451, .5]
        with tempfile.TemporaryDirectory() as tmp:
            old_dir, new_dir = Path(tmp) / "old", Path(tmp) / "new"
            old_dir.mkdir()
            new_dir.mkdir()
            old.to_csv(old_dir / "nodes_roles.csv", index=False)
            new.to_csv(new_dir / "nodes_roles.csv", index=False)
            result = compare_outputs(old_dir, new_dir)
        self.assertEqual(result.gid.tolist(), ["1", "2"])
        self.assertEqual(result.priority_delta.tolist(), [.05, -.05])
        self.assertEqual(result.status.tolist(), ["kept", "kept"])
        self.assertFalse(result[["role_changed", "cluster_changed", "attention_changed"]].any().any())


if __name__ == "__main__":
    unittest.main()

