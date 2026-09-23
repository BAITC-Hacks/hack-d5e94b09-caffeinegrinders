"""Regression checks for seed component footprint."""

import unittest
import pandas as pd
import networkx as nx
from run import seed_component_summary


class RegressionTest(unittest.TestCase):

    def test_component_membership_does_not_imply_directed_reachability(self):
        graph = nx.DiGraph([(1, 2), (3, 2)])
        graph.add_node(4)
        df = pd.DataFrame({"gid": [4, 3, 2, 1], "is_seed": [True, True, False, True],
                           "cluster_id": [40, 30, 20, 10]})
        edges = pd.DataFrame([(1, 2, 100.0), (3, 2, 200.0)],
                             columns=["src", "dst", "sum_kzt"])
        result = seed_component_summary(graph, df, edges)
        self.assertEqual(result.seed_gid.tolist(), [1, 3, 4])
        self.assertEqual(result.component_id.tolist(), [1, 1, 2])
        self.assertEqual(result.component_nodes.tolist(), [3, 3, 1])
        self.assertEqual(result.component_seed.tolist(), [2, 2, 1])
        self.assertEqual(result.component_internal_kzt.tolist(), [300, 300, 0])
        self.assertEqual(result.component_edges.tolist(), [2, 2, 0])
        self.assertEqual(result.reachable_nodes.tolist(), [1, 1, 0])
        self.assertEqual(result.max_depth_reached.tolist(), [1, 1, 0])
        self.assertEqual(result.cluster_id.tolist(), [10, 30, 40])


if __name__ == "__main__":
    unittest.main()

