"""Regression checks for cluster attention flags."""

import unittest
import pandas as pd
from run import cluster_attention_summary


class RegressionTest(unittest.TestCase):

    def test_different_flags_share_nodes_without_inflating_cluster_population(self):
        df = pd.DataFrame({"gid": [1, 2, 3, 4], "cluster_id": [10, 10, 10, 20]})
        flags = pd.DataFrame([
            (2, 10, "cycle", .8, "cycle meaning"),
            (1, 10, "cycle", .8, "cycle meaning"),
            (1, 10, "relay_route", .8, "relay meaning"),
            (4, 20, "cycle", .3, "cycle meaning"),
        ], columns=["gid", "cluster_id", "flag", "priority_score", "meaning"])
        result = cluster_attention_summary(df, flags).set_index(["cluster_id", "flag"])
        self.assertEqual(len(result), 3)
        row = result.loc[(10, "cycle")]
        self.assertEqual(row.n_nodes, 2)
        self.assertEqual(row.share_cluster, .6667)
        self.assertEqual(row.top_gids, "1;2")
        self.assertEqual(row.meaning, "cycle meaning")
        self.assertEqual(result.loc[(10, "relay_route")].share_cluster, .3333)
        self.assertEqual(result.loc[(20, "cycle")].share_cluster, 1.0)


if __name__ == "__main__":
    unittest.main()

