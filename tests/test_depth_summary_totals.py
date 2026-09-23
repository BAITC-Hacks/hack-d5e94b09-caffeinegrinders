"""Regression coverage for depth summary totals."""

import unittest

import pandas as pd

from run import depth_summary


class SummaryRegressionTest(unittest.TestCase):

    def test_depth_summary_keeps_boundary_counts_and_sorts_tied_leaders(self):
        df = pd.DataFrame({
            "gid": [3, 1, 2], "depth": [4, 0, 4],
            "is_seed": [False, True, False], "boundary": [True, False, True],
            "in_kzt": [100, 0, 200], "out_kzt": [0, 300, 0],
            "priority_score": [.5, .9, .5], "attention": ["flag", "нет", "нет"],
        })
        result = depth_summary(df).set_index("depth")
        self.assertEqual(result.index.tolist(), [0, 4])
        self.assertEqual(result.n_nodes.tolist(), [1, 2])
        self.assertEqual(result.share_nodes.tolist(), [.3333, .6667])
        self.assertEqual(result.n_seed.tolist(), [1, 0])
        self.assertEqual(result.boundary_nodes.tolist(), [0, 2])
        self.assertEqual(result.attention_nodes.tolist(), [0, 1])
        self.assertEqual(result.in_kzt.tolist(), [0, 300])
        self.assertEqual(result.out_kzt.tolist(), [300, 0])
        self.assertEqual(result.loc[4].top_gids, "2;3")
        self.assertEqual(result.loc[4].avg_priority_score, .5)


if __name__ == "__main__":
    unittest.main()

