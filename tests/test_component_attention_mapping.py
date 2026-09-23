"""Regression checks for component attention mapping."""

import unittest
import pandas as pd
import networkx as nx
from run import component_attention_summary


class RegressionTest(unittest.TestCase):

    def test_flags_group_by_weak_component_instead_of_cluster(self):
        graph = nx.DiGraph([(1, 2), (3, 2), (4, 5)])
        df = pd.DataFrame({"gid": [1, 2, 3, 4, 5], "cluster_id": [10, 20, 30, 10, 20]})
        flags = pd.DataFrame([
            (1, 10, "cycle", .8, "cycle"),
            (3, 30, "cycle", .4, "cycle"),
            (4, 10, "cycle", .6, "cycle"),
            (3, 30, "relay_route", .4, "relay"),
        ], columns=["gid", "cluster_id", "flag", "priority_score", "meaning"])
        original = flags.copy(deep=True)
        result = component_attention_summary(graph, df, flags).set_index(["component_id", "flag"])
        self.assertEqual(len(result), 3)
        first = result.loc[(1, "cycle")]
        self.assertEqual(first.n_nodes, 2)
        self.assertEqual(first.share_component, .6667)
        self.assertEqual(first.avg_priority_score, .6)
        self.assertEqual(first.top_gids, "1;3")
        self.assertEqual(result.loc[(2, "cycle")].share_component, .5)
        self.assertEqual(result.loc[(1, "relay_route")].share_component, .3333)
        pd.testing.assert_frame_equal(flags, original)


if __name__ == "__main__":
    unittest.main()

