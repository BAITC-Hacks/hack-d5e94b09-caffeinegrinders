"""Regression coverage for cluster role shares."""

import unittest

import pandas as pd

from run import cluster_role_matrix


class SummaryRegressionTest(unittest.TestCase):

    def test_role_shares_use_cluster_size_and_leaders_break_ties_by_gid(self):
        df = pd.DataFrame({
            "gid": [3, 1, 2, 4], "cluster_id": [10, 10, 10, 20],
            "role": ["transit", "transit", "terminal", "terminal"],
            "in_kzt": [100, 200, 300, 400], "out_kzt": [90, 180, 0, 0],
            "priority_score": [.8, .8, .2, .4],
        })
        result = cluster_role_matrix(df).set_index(["cluster_id", "role"])
        self.assertEqual(set(result.index), {(10, "transit"), (10, "terminal"), (20, "terminal")})
        row = result.loc[(10, "transit")]
        self.assertEqual(row.n_nodes, 2)
        self.assertEqual(row.share_cluster, .6667)
        self.assertEqual(row.in_kzt, 300)
        self.assertEqual(row.out_kzt, 270)
        self.assertEqual(row.avg_priority_score, .8)
        self.assertEqual(row.top_gids, "1;3")
        self.assertEqual(result.loc[(10, "terminal")].share_cluster, .3333)
        self.assertEqual(result.loc[(20, "terminal")].share_cluster, 1.0)


if __name__ == "__main__":
    unittest.main()

