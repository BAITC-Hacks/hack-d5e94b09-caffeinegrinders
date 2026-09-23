"""Regression coverage for cycle membership closure."""

import unittest

import pandas as pd

from run import cycle_node_membership


class SummaryRegressionTest(unittest.TestCase):

    def test_closing_node_is_not_duplicated_and_self_loop_has_one_member(self):
        df = pd.DataFrame({
            "gid": [3, 1, 2], "role": ["terminal", "distributor", "transit"],
            "cluster_id": [3, 1, 2], "is_seed": [False, True, False],
            "priority_score": [.3, .1, .2],
        })
        cycles = pd.DataFrame([
            {"cycle_id": 5, "path": "1 → 2 → 3 → 1", "bottleneck_kzt": 100.0},
            {"cycle_id": 6, "path": "2 → 2", "bottleneck_kzt": 50.0},
        ])
        result = cycle_node_membership(cycles, df)
        self.assertEqual(result.gid.tolist(), [1, 2, 3, 2])
        self.assertEqual(result.position.tolist(), [1, 2, 3, 1])
        self.assertEqual(result.cycle_id.tolist(), [5, 5, 5, 6])
        self.assertEqual(result.bottleneck_kzt.tolist(), [100, 100, 100, 50])
        self.assertEqual(result.role.tolist(), ["distributor", "transit", "terminal", "transit"])
        self.assertEqual(result.is_seed.tolist(), [True, False, False, False])
        self.assertEqual(result.path.tolist(), ["1 → 2 → 3 → 1"] * 3 + ["2 → 2"])


if __name__ == "__main__":
    unittest.main()

