"""Seed coverage follows outgoing transfers for at most four hops."""

import unittest

import networkx as nx
import pandas as pd

from run import seed_coverage


class SeedCoverageLimitsTest(unittest.TestCase):
    def test_coverage_respects_direction_depth_and_isolated_seeds(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(1, 9))
        # Node 6 is five hops away; node 7 only pays the seed; node 8 is isolated.
        for src, dst in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 1)]:
            graph.add_edge(src, dst, sum_kzt=100.0)
        nodes = pd.DataFrame({
            "gid": list(range(1, 9)), "cluster_id": [10] * 7 + [20],
            "is_seed": [True, False, False, False, False, False, False, True],
        })
        report = seed_coverage(graph, nodes)
        self.assertEqual(report.to_dict("records"), [
            {"seed_gid": 1, "cluster_id": 10, "reachable_nodes": 4,
             "direct_recipients": 1, "direct_out_kzt": 100.0, "max_depth_reached": 4},
            {"seed_gid": 8, "cluster_id": 20, "reachable_nodes": 0,
             "direct_recipients": 0, "direct_out_kzt": 0.0, "max_depth_reached": 0},
        ])


if __name__ == "__main__":
    unittest.main()
