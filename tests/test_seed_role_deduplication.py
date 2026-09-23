"""Regression coverage for seed role deduplication."""

import unittest

import pandas as pd
import networkx as nx

from run import seed_role_reach


class SummaryRegressionTest(unittest.TestCase):

    def test_role_reach_counts_shared_descendant_once_and_excludes_seed(self):
        graph = nx.DiGraph([(1, 2), (1, 3), (2, 4), (3, 4), (4, 1), (5, 1)])
        df = pd.DataFrame({
            "gid": [1, 2, 3, 4, 5], "cluster_id": [1] * 5,
            "is_seed": [True, False, False, False, False],
            "role": ["distributor", "transit", "transit", "terminal", "coordinator"],
            "boundary": [False, False, False, True, False],
            "attention": ["нет", "нет", "flag", "flag", "нет"],
            "priority_score": [1.0, .5, .5, .9, 1.0],
        })
        result = seed_role_reach(graph, df)
        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row.reachable_nodes, 3)
        self.assertEqual(row.max_depth_reached, 2)
        self.assertEqual(row.n_transit, 2)
        self.assertEqual(row.n_terminal, 1)
        self.assertEqual(row.n_distributor, 0)
        self.assertEqual(row.n_coordinator, 0)
        self.assertEqual(row.boundary_nodes, 1)
        self.assertEqual(row.attention_nodes, 2)
        self.assertEqual(row.top_reachable_gids, "4;2;3")


if __name__ == "__main__":
    unittest.main()

