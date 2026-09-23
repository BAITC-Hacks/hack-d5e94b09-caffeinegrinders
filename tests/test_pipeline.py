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

    def test_priority_breakdown_matches_score(self):
        pieces = ["priority_payers", "priority_incoming", "priority_seed_reach",
                  "priority_bridge", "priority_recipients", "priority_rapid"]
        self.assertTrue(self.nodes[pieces + ["priority_base", "priority_role_factor",
                                             "priority_boundary_factor", "priority_seed_factor"]].notna().all().all())
        expected_base = self.nodes[pieces].sum(axis=1).round(6)
        self.assertTrue(np.allclose(self.nodes.priority_base, expected_base, atol=.00001))
        expected_score = (self.nodes.priority_base
                          * self.nodes.priority_role_factor
                          * self.nodes.priority_boundary_factor
                          * self.nodes.priority_seed_factor).clip(0, 1).round(6)
        self.assertTrue(np.allclose(self.nodes.priority_score, expected_score, atol=.00001))
        self.assertTrue(self.nodes.priority_role_factor.between(.7, 1).all())
        self.assertTrue(self.nodes.priority_boundary_factor.isin([.65, 1.0]).all())
        self.assertTrue(self.nodes.priority_seed_factor.isin([.85, 1.0]).all())

    def test_cycles_follow_real_transfers(self):
        cycles = pd.read_csv(self.out / "cycles.csv")
        edges = pd.read_parquet(ROOT / "data" / "edges.parquet")
        amount = {(e.src, e.dst): e.sum_kzt for e in edges.itertuples(index=False)}
        seeds = set(self.nodes.gid[self.nodes.is_seed])
        for row in cycles.itertuples(index=False):
            path = [int(x) for x in row.path.split(" → ")]
            self.assertEqual(path[0], path[-1])
            self.assertEqual(row.length, len(path) - 1)
            self.assertLessEqual(row.length, 4)
            hops = list(zip(path, path[1:]))
            self.assertTrue(all(hop in amount for hop in hops))
            self.assertTrue(np.isclose(row.bottleneck_kzt, min(amount[hop] for hop in hops), atol=.01))
            self.assertEqual(row.n_seed, len(set(path) & seeds))
        in_cycles = set(int(x) for path in cycles.path for x in path.split(" → "))
        self.assertEqual(in_cycles, set(self.nodes.gid[self.nodes.cycles > 0]))

    def test_boundary_estimate_and_flags(self):
        boundary = self.nodes[self.nodes.boundary]
        self.assertTrue(boundary.p_hidden_outgoing.between(0, 1).all())
        self.assertTrue((self.nodes.p_hidden_outgoing[~self.nodes.boundary] == 0).all())
        self.assertTrue(self.nodes.attention.str.len().between(1, 200).all())

    def test_resilience_and_gaps(self):
        resilience = pd.read_csv(self.out / "resilience.csv")
        self.assertEqual(set(resilience.strategy), {"priority", "random"})
        start = resilience[resilience.removed == 0]
        self.assertEqual(start.seed_reach_share.tolist(), [1.0, 1.0])
        for _, group in resilience.groupby("strategy"):
            self.assertTrue(group.sort_values("removed").edges_left.is_monotonic_decreasing)
        gaps = pd.read_csv(self.out / "data_gaps.csv").set_index("gap")
        boundary_total = gaps.loc[["boundary_likely_continues", "boundary_other"], "n_nodes"].sum()
        self.assertEqual(boundary_total, self.nodes.boundary.sum())
        self.assertEqual(gaps.loc["seed_without_outgoing", "n_nodes"],
                         (self.nodes.is_seed & (self.nodes.out_deg == 0)).sum())
        self.assertTrue(gaps.next_request.str.len().gt(0).all())

    def test_timeline_matches_transactions(self):
        timeline = pd.read_csv(self.out / "timeline.csv")
        tx = pd.read_parquet(ROOT / "data" / "transactions.parquet")
        tx["date"] = pd.to_datetime(tx.date).dt.date.astype(str)
        self.assertEqual(
            round(timeline.in_kzt.sum(), 2),
            round(tx.sum_kzt.sum(), 2),
        )
        self.assertEqual(
            round(timeline.out_kzt.sum(), 2),
            round(tx.sum_kzt.sum(), 2),
        )
        busiest = timeline.sort_values(["in_tx", "out_tx"], ascending=False).iloc[0]
        incoming = tx[(tx.dst == busiest.gid) & (tx.date == busiest.date)]
        outgoing = tx[(tx.src == busiest.gid) & (tx.date == busiest.date)]
        self.assertEqual(busiest.in_tx, len(incoming))
        self.assertEqual(busiest.out_tx, len(outgoing))
        self.assertEqual(busiest.unique_payers, incoming.src.nunique())
        self.assertEqual(busiest.unique_recipients, outgoing.dst.nunique())

    def test_viewer_contains_graph(self):
        html = (self.out / "network.html").read_text(encoding="utf-8")
        self.assertNotIn("/* GRAPH_DATA */ null", html)
        self.assertIn(str(self.top.iloc[0].gid), html)
        self.assertIn('"timeline":[', html)
        self.assertIn("<canvas", html)

    def test_outputs_are_identical_after_input_rows_are_shuffled(self):
        """Input row order must not change scores, communities or exported bytes."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            out_dir = Path(tmp) / "out"
            data_dir.mkdir()
            for name in ("nodes", "edges", "transactions"):
                source = pd.read_parquet(ROOT / "data" / f"{name}.parquet")
                shuffled = source.sample(frac=1, random_state=2026).reset_index(drop=True)
                shuffled.to_parquet(data_dir / f"{name}.parquet", index=False)
            result = subprocess.run(
                [sys.executable, str(ROOT / "run.py"),
                 "--data", str(data_dir), "--out", str(out_dir)],
                capture_output=True, text=True, timeout=300,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for name in ("nodes_roles.csv", "clusters.csv", "top_nodes.csv",
                         "cycles.csv", "resilience.csv", "data_gaps.csv", "timeline.csv", "network.html"):
                with self.subTest(file=name):
                    self.assertTrue((out_dir / name).is_file(), f"Missing output: {name}")
                    self.assertEqual((self.out / name).read_bytes(),
                                     (out_dir / name).read_bytes(),
                                     f"Shuffling input rows changed {name}")


if __name__ == "__main__":
    unittest.main()
