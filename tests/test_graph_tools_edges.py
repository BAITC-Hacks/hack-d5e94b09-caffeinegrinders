"""Edge cases of the assistant graph tools: invalid input and honest truncation."""

import unittest
from pathlib import Path

import networkx as nx
import pandas as pd

import graph_tools
from graph_tools import GraphIndex

ROOT = Path(__file__).resolve().parents[1]


class GraphToolsEdgeCaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = GraphIndex(ROOT / "data")
        cls.hub = int(cls.index.top_nodes(limit=1)["shown"][0]["gid"])

    def test_same_src_and_dst_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "node_cycles"):
            self.index.money_paths(self.hub, self.hub)

    def test_invalid_direction_is_rejected_everywhere(self):
        with self.assertRaisesRegex(ValueError, "direction"):
            self.index.counterparties(self.hub, direction="sideways")
        with self.assertRaisesRegex(ValueError, "direction"):
            self.index.shared_counterparties([self.hub], direction="sideways")

    def test_truncated_path_scan_is_reported(self):
        # Any destination with at least two distinct routes from the hub.
        dst = next(node for node, hops in
                   nx.single_source_shortest_path_length(self.index.graph, self.hub, cutoff=2).items()
                   if node != self.hub
                   and sum(1 for _ in nx.all_simple_paths(self.index.graph, self.hub, node, cutoff=3)) >= 2)
        original = graph_tools.PATH_SCAN_LIMIT
        graph_tools.PATH_SCAN_LIMIT = 1
        try:
            result = self.index.money_paths(self.hub, dst, max_hops=3)
        finally:
            graph_tools.PATH_SCAN_LIMIT = original
        self.assertTrue(result["truncated"])
        self.assertEqual(result["total"], 1)

    def test_full_path_scan_is_exact(self):
        edge = next(self.index.edges.itertuples(index=False))
        result = self.index.money_paths(int(edge.src), int(edge.dst), max_hops=1)
        self.assertFalse(result["truncated"])
        self.assertEqual(result["total"], 1)


class FeatureBoundsTest(unittest.TestCase):
    def test_near_threshold_counts_only_the_documented_band(self):
        nodes = pd.read_csv(ROOT / "out" / "nodes_roles.csv")
        tx = pd.read_parquet(ROOT / "data" / "transactions.parquet")
        band = tx[(tx.sum_kzt >= 5_000) & (tx.sum_kzt < 10_000)].groupby("dst").size()
        expected = nodes.gid.map(band).fillna(0).astype(int)
        self.assertTrue((nodes.near_threshold_in == expected).all())

    def test_resilience_never_reports_removing_more_nodes_than_exist(self):
        resilience = pd.read_csv(ROOT / "out" / "resilience.csv")
        n_nodes = len(pd.read_parquet(ROOT / "data" / "nodes.parquet"))
        self.assertTrue((resilience.removed <= n_nodes).all())
        self.assertTrue(resilience.groupby("strategy").removed.apply(lambda s: s.is_unique).all())


if __name__ == "__main__":
    unittest.main()
