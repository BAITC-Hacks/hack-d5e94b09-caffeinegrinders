"""Regression checks for role depth denominators."""

import unittest
import pandas as pd
from run import role_depth_matrix


class RegressionTest(unittest.TestCase):

    def test_depth_shares_use_role_population_and_preserve_role_order(self):
        df = pd.DataFrame({
            "gid": [4, 3, 2, 1], "role": ["peripheral", "transit", "transit", "transit"],
            "depth": [4, 2, 2, 1], "is_seed": [False] * 4,
            "boundary": [True, False, False, False], "priority_score": [.1, .8, .8, .2],
        })
        result = role_depth_matrix(df)
        self.assertEqual(list(zip(result.role, result.depth)),
                         [("transit", 1), ("transit", 2), ("peripheral", 4)])
        self.assertEqual(result.share_role.tolist(), [.3333, .6667, 1.0])
        self.assertEqual(result.n_nodes.tolist(), [1, 2, 1])
        self.assertEqual(result.top_gids.tolist(), ["1", "2;3", "4"])
        self.assertEqual(result.boundary_nodes.tolist(), [0, 0, 1])
        self.assertEqual(result.avg_priority_score.tolist(), [.2, .8, .1])


if __name__ == "__main__":
    unittest.main()

