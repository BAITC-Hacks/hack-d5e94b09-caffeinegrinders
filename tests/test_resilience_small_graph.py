"""Regression checks for resilience small graph."""

import unittest
import pandas as pd
import networkx as nx
from run import network_resilience


class RegressionTest(unittest.TestCase):

    def test_removal_steps_are_capped_and_complete_removal_leaves_no_graph(self):
        graph = nx.DiGraph([(1, 2), (2, 3)])
        df = pd.DataFrame({"gid": [1, 2, 3], "is_seed": [True, False, False],
                           "priority_score": [.9, .5, .1]})
        result = network_resilience(graph, df)
        self.assertEqual(len(result), 4)
        self.assertEqual(set(result.strategy), {"priority", "random"})
        for strategy, group in result.groupby("strategy"):
            with self.subTest(strategy=strategy):
                self.assertEqual(group.removed.tolist(), [0, 3])
                self.assertEqual(group.edges_left.tolist(), [2, 0])
                self.assertEqual(group.seed_reach_pairs.tolist(), [2, 0])
                self.assertEqual(group.seed_reach_share.tolist(), [1.0, 0.0])
                self.assertEqual(group.largest_component.tolist(), [3, 0])
                self.assertEqual(group.linked_components.tolist(), [1, 0])
        self.assertEqual(set(graph.edges), {(1, 2), (2, 3)})


if __name__ == "__main__":
    unittest.main()

