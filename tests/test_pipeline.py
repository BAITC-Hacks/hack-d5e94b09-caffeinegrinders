"""End-to-end checks of the deliverables, independent of the role heuristics."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        subprocess.run([sys.executable, str(ROOT / "run.py"),
                        "--data", str(ROOT / "data"), "--out", str(cls.out)],
                       check=True, capture_output=True, text=True)
        cls.nodes = pd.read_csv(cls.out / "nodes_roles.csv")
        cls.clusters = pd.read_csv(cls.out / "clusters.csv")
        cls.top = pd.read_csv(cls.out / "top_nodes.csv")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_node_has_explainable_role(self):
        source = pd.read_parquet(ROOT / "data" / "nodes.parquet")
        self.assertEqual(len(self.nodes), len(source))
        self.assertEqual(set(self.nodes.gid), set(source.gid))
        self.assertTrue(self.nodes.gid.is_unique)
        self.assertTrue(set(self.nodes.role).issubset({"coordinator", "consolidator", "transit", "distributor", "terminal", "peripheral"}))
        self.assertTrue(self.nodes[["role", "role_score", "cluster_id", "priority_score", "evidence"]].notna().all().all())
        self.assertTrue(self.nodes.evidence.str.len().between(1, 200).all())
        self.assertTrue(self.nodes.role_score.between(0, 1).all())
        self.assertTrue(self.nodes.priority_score.between(0, 1).all())
        self.assertFalse(((self.nodes.depth == 4) & (self.nodes.out_deg == 0) & (self.nodes.role == "terminal")).any())
        self.assertFalse((self.nodes.is_seed & (self.nodes.role == "transit")).any())

    def test_clusters_and_top_are_consistent(self):
        self.assertEqual(set(self.clusters.cluster_id), set(self.nodes.cluster_id))
        sizes = self.nodes.groupby("cluster_id").size()
        seeds = self.nodes.groupby("cluster_id").is_seed.sum()
        for row in self.clusters.itertuples(index=False):
            self.assertEqual(row.n_nodes, sizes[row.cluster_id])
            self.assertEqual(row.n_seed, seeds[row.cluster_id])
            self.assertTrue(bool(row.hypothesis))
        edges = pd.read_parquet(ROOT / "data" / "edges.parquet")
        lookup = dict(zip(self.nodes.gid, self.nodes.cluster_id))
        totals = {}
        for edge in edges.itertuples(index=False):
            if lookup[edge.src] == lookup[edge.dst]:
                cid = lookup[edge.src]
                totals[cid] = totals.get(cid, 0) + edge.sum_kzt
        for row in self.clusters.itertuples(index=False):
            self.assertTrue(np.isclose(row.sum_kzt_internal, totals.get(row.cluster_id, 0), atol=.01))
        self.assertGreaterEqual(len(self.top), 20)
        self.assertEqual(self.top['rank'].tolist(), list(range(1, len(self.top) + 1)))
        self.assertTrue(self.top.priority_score.is_monotonic_decreasing)
        self.assertTrue(self.top.why.str.len().gt(0).all())

    def test_viewer_contains_graph(self):
        html = (self.out / "network.html").read_text(encoding="utf-8")
        self.assertNotIn("/* GRAPH_DATA */ null", html)
        self.assertIn(str(self.top.iloc[0].gid), html)
        self.assertIn("<canvas", html)


if __name__ == "__main__":
    unittest.main()
