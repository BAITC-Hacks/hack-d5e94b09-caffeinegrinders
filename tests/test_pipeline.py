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

    def test_routes_are_backed_by_transactions(self):
        routes = pd.read_csv(self.out / "routes.csv")
        self.assertGreater(len(routes), 0)
        self.assertTrue(set(routes.kind).issubset({"repeated", "chain"}))
        tx = pd.read_parquet(ROOT / "data" / "transactions.parquet")
        tx["date"] = pd.to_datetime(tx.date)
        edges = set(zip(tx.src, tx.dst))
        middles, inner = {}, {}
        for row in routes.itertuples(index=False):
            path = [int(x) for x in row.path.split(" → ")]
            self.assertEqual(row.hops, len(path) - 1)
            self.assertTrue(all(hop in edges for hop in zip(path, path[1:])))
            if row.kind == "chain":
                self.assertEqual(row.hops, 3)
                for node in path[1:-1]:
                    inner[node] = inner.get(node, 0) + 1
                continue
            a, b, c = path
            ins = tx[(tx.src == a) & (tx.dst == b)]
            outs = tx[(tx.src == b) & (tx.dst == c)]
            relayed = [out for out in outs.itertuples() if any(
                0 <= (out.date - i.date).days <= 2
                and .5 * i.sum_kzt <= out.sum_kzt <= 1.05 * i.sum_kzt
                for i in ins.itertuples())]
            self.assertGreaterEqual(row.relay_days, 2)
            self.assertEqual(row.relay_days, len({out.date for out in relayed}))
            self.assertTrue(np.isclose(row.forwarded_kzt, sum(out.sum_kzt for out in relayed), atol=.01))
            middles[b] = middles.get(b, 0) + 1
        lookup = self.nodes.set_index("gid")
        for gid, count in middles.items():
            self.assertEqual(lookup.relay_routes[gid], count)
        for gid, count in inner.items():
            self.assertEqual(lookup.chain_transits[gid], count)
        route_nodes = pd.read_csv(self.out / "route_nodes.csv")
        self.assertEqual(route_nodes.route_id.nunique(), len(routes))
        by_route = route_nodes.groupby("route_id").size()
        for row in routes.set_index("route_id").itertuples():
            self.assertEqual(by_route.loc[row.Index], row.hops + 1)
        self.assertTrue(set(route_nodes.gid).issubset(set(self.nodes.gid)))
        self.assertTrue(route_nodes.position.ge(1).all())

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
        flags = pd.read_csv(self.out / "risk_flags.csv").set_index("flag")
        self.assertEqual(flags.loc["cycle", "n_nodes"], (self.nodes.cycles > 0).sum())
        self.assertEqual(flags.loc["relay_route", "n_nodes"], (self.nodes.relay_routes > 0).sum())
        self.assertTrue(flags.share.between(0, 1).all())
        attention_examples = pd.read_csv(self.out / "attention_examples.csv")
        counts = attention_examples.groupby("flag").gid.nunique().to_dict()
        for flag, row in flags.iterrows():
            self.assertEqual(counts.get(flag, 0), row.n_nodes)
        self.assertTrue(attention_examples.metric_value.notna().all())
        self.assertTrue(attention_examples.evidence.str.len().between(1, 200).all())
        flows = pd.read_csv(self.out / "cluster_flows.csv")
        lookup = dict(zip(self.nodes.gid, self.nodes.cluster_id))
        expected = 0.0
        for edge in pd.read_parquet(ROOT / "data" / "edges.parquet").itertuples(index=False):
            if lookup[edge.src] != lookup[edge.dst]:
                expected += edge.sum_kzt
        self.assertTrue(np.isclose(flows.sum_kzt.sum(), expected, atol=.01))
        seeds = pd.read_csv(self.out / "seed_coverage.csv")
        self.assertEqual(len(seeds), int(self.nodes.is_seed.sum()))
        self.assertTrue(seeds.max_depth_reached.between(0, 4).all())
        self.assertTrue(seeds.reachable_nodes.ge(0).all())
        components = pd.read_csv(self.out / "components.csv")
        self.assertEqual(components.n_nodes.sum(), len(self.nodes))
        self.assertEqual(components.is_main_component.sum(), 1)
        self.assertTrue(components.n_nodes.is_monotonic_decreasing)
        amount_bands = pd.read_csv(self.out / "amount_bands.csv")
        tx = pd.read_parquet(ROOT / "data" / "transactions.parquet")
        self.assertEqual(amount_bands.n_tx.sum(), len(tx))
        self.assertTrue(np.isclose(amount_bands.sum_kzt.sum(), tx.sum_kzt.sum(), atol=.01))
        self.assertTrue(amount_bands.share_tx.between(0, 1).all())
        self.assertTrue(amount_bands.share_kzt.between(0, 1).all())
        roles = pd.read_csv(self.out / "role_summary.csv").set_index("role")
        self.assertEqual(set(roles.index), {"coordinator", "consolidator", "distributor", "transit", "terminal", "peripheral"})
        self.assertEqual(roles.n_nodes.sum(), len(self.nodes))
        self.assertTrue(np.isclose(roles.share_nodes.sum(), 1.0, atol=.001))
        for role, count in self.nodes.role.value_counts().items():
            self.assertEqual(roles.loc[role, "n_nodes"], count)
        top_edges = pd.read_csv(self.out / "top_edges.csv")
        source_edges = pd.read_parquet(ROOT / "data" / "edges.parquet")
        self.assertLessEqual(len(top_edges), 200)
        self.assertEqual(top_edges["rank"].tolist(), list(range(1, len(top_edges) + 1)))
        self.assertTrue(top_edges.sum_kzt.is_monotonic_decreasing)
        self.assertTrue(np.isclose(top_edges.iloc[0].sum_kzt, source_edges.sum_kzt.max(), atol=.01))
        self.assertTrue(set(zip(top_edges.src, top_edges.dst)).issubset(set(zip(source_edges.src, source_edges.dst))))
        daily = pd.read_csv(self.out / "daily_summary.csv")
        tx["date"] = pd.to_datetime(tx.date).dt.date.astype(str)
        self.assertEqual(daily.n_tx.sum(), len(tx))
        self.assertTrue(np.isclose(daily.sum_kzt.sum(), tx.sum_kzt.sum(), atol=.01))
        self.assertTrue(daily.date.is_monotonic_increasing)
        busiest = daily.sort_values(["n_tx", "sum_kzt"], ascending=False).iloc[0]
        day_tx = tx[tx.date == busiest.date]
        self.assertEqual(busiest.unique_senders, day_tx.src.nunique())
        self.assertEqual(busiest.unique_recipients, day_tx.dst.nunique())
        boundary = pd.read_csv(self.out / "boundary_review.csv")
        self.assertEqual(len(boundary), int(self.nodes.boundary.sum()))
        self.assertEqual(boundary["rank"].tolist(), list(range(1, len(boundary) + 1)))
        self.assertTrue(boundary.p_hidden_outgoing.is_monotonic_decreasing)
        self.assertTrue((boundary.depth == 4).all())
        self.assertTrue(boundary.next_request.str.contains("Запросить исходящие").all())
        cluster_roles = pd.read_csv(self.out / "cluster_roles.csv")
        by_cluster = cluster_roles.groupby("cluster_id").n_nodes.sum().sort_index()
        expected_by_cluster = self.nodes.groupby("cluster_id").size().sort_index()
        self.assertEqual(by_cluster.to_dict(), expected_by_cluster.to_dict())
        self.assertTrue(cluster_roles.share_cluster.between(0, 1).all())
        self.assertTrue(set(cluster_roles.role).issubset(set(self.nodes.role)))
        seed_roles = pd.read_csv(self.out / "seed_role_reach.csv")
        role_cols = ["n_coordinator", "n_consolidator", "n_distributor",
                     "n_transit", "n_terminal", "n_peripheral"]
        self.assertEqual(len(seed_roles), int(self.nodes.is_seed.sum()))
        coverage = seeds.set_index("seed_gid")
        for row in seed_roles.itertuples(index=False):
            self.assertEqual(row.reachable_nodes, coverage.loc[row.seed_gid, "reachable_nodes"])
        self.assertTrue((seed_roles[role_cols].sum(axis=1) == seed_roles.reachable_nodes).all())
        self.assertTrue(seed_roles.max_depth_reached.between(0, 4).all())

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
        self.assertIn('"cycles":[', html)
        self.assertIn('"clusters":[', html)
        self.assertIn('"clusterFlows":[', html)
        self.assertIn('"gaps":[', html)
        self.assertIn('"seedCoverage":[', html)
        self.assertIn('"components":[', html)
        self.assertIn('"amountBands":[', html)
        self.assertIn('"roleSummary":[', html)
        self.assertIn('"topEdges":[', html)
        self.assertIn('"dailySummary":[', html)
        self.assertIn('"boundaryReview":[', html)
        self.assertIn('"clusterRoles":[', html)
        self.assertIn('"seedRoleReach":[', html)
        self.assertIn('"attentionExamples":[', html)
        self.assertIn('"routeNodes":[', html)
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
                         "cycles.csv", "routes.csv", "resilience.csv", "data_gaps.csv",
                         "risk_flags.csv", "cluster_flows.csv", "seed_coverage.csv",
                         "components.csv", "amount_bands.csv", "role_summary.csv",
                         "top_edges.csv", "daily_summary.csv", "boundary_review.csv",
                         "cluster_roles.csv", "seed_role_reach.csv", "attention_examples.csv",
                         "route_nodes.csv", "timeline.csv", "network.html"):
                with self.subTest(file=name):
                    self.assertTrue((out_dir / name).is_file(), f"Missing output: {name}")
                    self.assertEqual((self.out / name).read_bytes(),
                                     (out_dir / name).read_bytes(),
                                     f"Shuffling input rows changed {name}")


if __name__ == "__main__":
    unittest.main()
