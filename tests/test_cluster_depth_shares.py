"""Regression coverage for cluster depth shares."""

import unittest

import pandas as pd

from run import cluster_depth_matrix


class SummaryRegressionTest(unittest.TestCase):

    def test_depth_shares_are_local_to_each_cluster(self):
        df = pd.DataFrame({
            "gid": [4, 3, 2, 1], "cluster_id": [20, 10, 10, 10],
            "depth": [4, 4, 4, 0], "is_seed": [False, False, False, True],
            "boundary": [True, True, True, False], "priority_score": [.2, .5, .5, .9],
        })
        result = cluster_depth_matrix(df)
        self.assertEqual(list(zip(result.cluster_id, result.depth)), [(10, 0), (10, 4), (20, 4)])
        self.assertEqual(result.n_nodes.tolist(), [1, 2, 1])
        self.assertEqual(result.share_cluster.tolist(), [.3333, .6667, 1.0])
        self.assertEqual(result.n_seed.tolist(), [1, 0, 0])
        self.assertEqual(result.boundary_nodes.tolist(), [0, 2, 1])
        self.assertEqual(result.top_gids.tolist(), ["1", "2;3", "4"])
        self.assertEqual(result.avg_priority_score.tolist(), [.9, .5, .2])


if __name__ == "__main__":
    unittest.main()

