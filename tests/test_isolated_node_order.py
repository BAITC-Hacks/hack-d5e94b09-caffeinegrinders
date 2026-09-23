"""Regression checks for isolated node order."""

import unittest
import pandas as pd
import networkx as nx
from run import isolated_nodes_report


class RegressionTest(unittest.TestCase):

    def test_isolated_report_excludes_one_way_connections_and_prioritises_seeds(self):
        graph = nx.DiGraph([(1, 2)])
        graph.add_nodes_from([3, 4, 5, 6])
        df = pd.DataFrame({
            "gid": [6, 5, 4, 3, 2, 1],
            "in_deg": [0, 0, 0, 0, 1, 0], "out_deg": [0, 0, 0, 0, 0, 1],
            "is_seed": [True, True, True, False, False, True],
            "priority_score": [.4, .4, .6, .99, 1.0, 1.0],
            "depth": [0, 0, 0, 4, 1, 0], "role": ["peripheral"] * 6,
            "cluster_id": [6, 5, 4, 3, 2, 1],
            "evidence": ["no visible links"] * 6, "attention": ["нет"] * 6,
        })
        result = isolated_nodes_report(graph, df)
        self.assertEqual(result.gid.tolist(), [4, 5, 6, 3])
        self.assertEqual(result.component_id.tolist(), [3, 4, 5, 2])
        self.assertEqual(result.is_seed.tolist(), [True, True, True, False])
        self.assertNotIn("component_id", df.columns)


if __name__ == "__main__":
    unittest.main()

