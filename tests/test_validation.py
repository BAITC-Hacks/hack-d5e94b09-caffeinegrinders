"""Regression checks for data that previously passed validation silently."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from run import (
    load_data, graph_features, assign_clusters, score_roles, rank_nodes,
    find_cycles, find_routes, flag_attention, network_resilience, data_gaps,
    risk_flags_summary, attention_examples, cluster_attention_summary,
    role_attention_summary, depth_attention_summary, attention_overlap_summary,
    cluster_flows, cluster_flow_summary, seed_coverage, component_summary,
    seed_component_summary, component_role_matrix, component_attention_summary,
    isolated_nodes_report, amount_band_summary, role_summary, top_edge_summary, daily_summary,
    boundary_review, cluster_role_matrix, seed_role_reach, route_node_membership,
    cycle_node_membership, depth_summary, cluster_depth_matrix, role_depth_matrix,
    write_outputs, daily_timeline, role_flows, depth_flows, seed_overlap,
    seed_attention_summary, top_counterparties,
)


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.nodes = pd.DataFrame({"gid": [1, 2], "depth": [0, 1], "is_seed": [True, False]})
        self.edges = pd.DataFrame({"src": [1], "dst": [2], "sum_kzt": [1_000_000.0], "n_tx": [1]})
        self.tx = pd.DataFrame({"src": [1], "dst": [2], "date": ["2026-07-01"], "sum_kzt": [1_000_000.0]})

    def load(self):
        for name, frame in (("nodes", self.nodes), ("edges", self.edges), ("transactions", self.tx)):
            frame.to_parquet(self.path / f"{name}.parquet", index=False)
        return load_data(self.path)

    def test_valid_data(self):
        nodes, edges, tx = self.load()
        self.assertEqual(len(nodes), 2)
        self.assertEqual(edges.sum_kzt.sum(), tx.sum_kzt.sum())

    def test_missing_transaction_endpoint_is_not_silently_dropped(self):
        extra = self.tx.copy()
        extra["src"] = pd.Series([None], dtype="Int64")
        self.tx = pd.concat([self.tx, extra], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "null"):
            self.load()

    def test_missing_date(self):
        for missing in (None, "NaT", ""):
            with self.subTest(missing=missing):
                self.tx["date"] = [missing]
                with self.assertRaises(ValueError):
                    self.load()

    def test_large_amount_does_not_hide_disagreement(self):
        self.edges.loc[0, "sum_kzt"] += 5
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.load()

    def test_nonfinite_or_negative_amounts(self):
        for amount in (float("inf"), float("nan"), -10.0):
            with self.subTest(amount=amount):
                self.edges["sum_kzt"] = [amount]
                self.tx["sum_kzt"] = [amount]
                with self.assertRaises(ValueError):
                    self.load()

    def test_fractional_ids_are_not_truncated(self):
        self.nodes["gid"] = [1.5, 2.0]
        self.edges["src"] = [1.5]
        self.tx["src"] = [1.5]
        with self.assertRaisesRegex(ValueError, "integer"):
            self.load()

    def test_seed_flags_must_be_boolean(self):
        self.nodes["is_seed"] = ["True", "False"]
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.load()

    def test_empty_nodes_have_clear_error(self):
        self.nodes = self.nodes.iloc[:0]
        with self.assertRaisesRegex(ValueError, "empty"):
            self.load()

    def test_isolated_nodes_produce_all_outputs(self):
        self.edges = self.edges.iloc[:0]
        self.tx = self.tx.iloc[:0]
        nodes, edges, tx = self.load()
        graph, df = graph_features(nodes, edges, tx)
        df, cycles = find_cycles(graph, df)
        df, routes = find_routes(tx, df)
        df = flag_attention(rank_nodes(score_roles(assign_clusters(graph, df, edges))))
        resilience = network_resilience(graph, df)
        gaps = data_gaps(df, graph)
        risk_flags = risk_flags_summary(df)
        flows = cluster_flows(df, edges)
        cluster_flow_rows = cluster_flow_summary(df, edges)
        seed_report = seed_coverage(graph, df)
        seed_components = seed_component_summary(graph, df, edges)
        components = component_summary(graph, df, edges)
        component_roles = component_role_matrix(graph, df)
        isolated_nodes = isolated_nodes_report(graph, df)
        amount_bands = amount_band_summary(tx)
        roles = role_summary(df)
        top_edges = top_edge_summary(edges, df)
        daily = daily_summary(tx)
        boundary_queue = boundary_review(df)
        cluster_roles = cluster_role_matrix(df)
        seed_roles = seed_role_reach(graph, df)
        seed_overlap_rows = seed_overlap(graph, df)
        attention_rows = attention_examples(df)
        seed_attention = seed_attention_summary(graph, df, attention_rows)
        component_attention = component_attention_summary(graph, df, attention_rows)
        cluster_attention = cluster_attention_summary(df, attention_rows)
        role_attention = role_attention_summary(df, attention_rows)
        depth_attention = depth_attention_summary(df, attention_rows)
        attention_overlap = attention_overlap_summary(attention_rows)
        route_nodes = route_node_membership(routes, df)
        cycle_nodes = cycle_node_membership(cycles, df)
        depths = depth_summary(df)
        cluster_depths = cluster_depth_matrix(df)
        role_depths = role_depth_matrix(df)
        role_flow_rows = role_flows(edges, df)
        depth_flow_rows = depth_flows(edges, df)
        counterparties = top_counterparties(edges, df)
        timeline = daily_timeline(tx)
        write_outputs(df, edges, cycles, routes, resilience, gaps, risk_flags, flows,
                      cluster_flow_rows,
                      seed_report, seed_components, components, component_roles,
                      component_attention, isolated_nodes, amount_bands, roles, top_edges,
                      daily, boundary_queue, cluster_roles, seed_roles, seed_overlap_rows,
                      seed_attention,
                      attention_rows, cluster_attention, role_attention, depth_attention,
                      attention_overlap,
                      route_nodes, cycle_nodes, depths, cluster_depths, role_depths,
                      role_flow_rows, depth_flow_rows, counterparties,
                      timeline, self.path / "out")
        self.assertEqual(df.cluster_id.nunique(), 2)
        self.assertEqual(len(pd.read_csv(self.path / "out" / "top_nodes.csv")), 2)
        self.assertTrue(pd.read_csv(self.path / "out" / "cycles.csv").empty)
        self.assertTrue(pd.read_csv(self.path / "out" / "routes.csv").empty)
        self.assertTrue(pd.read_csv(self.path / "out" / "timeline.csv").empty)
        self.assertEqual(len(pd.read_csv(self.path / "out" / "risk_flags.csv")), 8)
        self.assertTrue(pd.read_csv(self.path / "out" / "cluster_flows.csv").empty)
        cluster_flow_rows = pd.read_csv(self.path / "out" / "cluster_flow_summary.csv")
        self.assertEqual(cluster_flow_rows.n_nodes.sum(), 2)
        self.assertEqual(cluster_flow_rows.cross_in_kzt.sum(), 0)
        self.assertEqual(cluster_flow_rows.cross_out_kzt.sum(), 0)
        self.assertEqual(len(pd.read_csv(self.path / "out" / "seed_coverage.csv")), 1)
        self.assertEqual(len(pd.read_csv(self.path / "out" / "seed_components.csv")), 1)
        self.assertEqual(len(pd.read_csv(self.path / "out" / "components.csv")), 2)
        self.assertEqual(pd.read_csv(self.path / "out" / "component_roles.csv").n_nodes.sum(), 2)
        self.assertTrue(pd.read_csv(self.path / "out" / "component_attention.csv").empty)
        self.assertEqual(len(pd.read_csv(self.path / "out" / "isolated_nodes.csv")), 2)
        self.assertTrue(pd.read_csv(self.path / "out" / "amount_bands.csv").empty)
        self.assertEqual(len(pd.read_csv(self.path / "out" / "role_summary.csv")), 6)
        self.assertTrue(pd.read_csv(self.path / "out" / "top_edges.csv").empty)
        self.assertTrue(pd.read_csv(self.path / "out" / "daily_summary.csv").empty)
        self.assertTrue(pd.read_csv(self.path / "out" / "boundary_review.csv").empty)
        self.assertEqual(pd.read_csv(self.path / "out" / "cluster_roles.csv").n_nodes.sum(), 2)
        seed_roles = pd.read_csv(self.path / "out" / "seed_role_reach.csv")
        self.assertEqual(len(seed_roles), 1)
        self.assertEqual(seed_roles.loc[0, "reachable_nodes"], 0)
        self.assertTrue(pd.read_csv(self.path / "out" / "seed_overlap.csv").empty)
        self.assertTrue(pd.read_csv(self.path / "out" / "seed_attention.csv").empty)
        self.assertTrue(pd.read_csv(self.path / "out" / "attention_examples.csv").empty)
        self.assertTrue(pd.read_csv(self.path / "out" / "cluster_attention.csv").empty)
        self.assertTrue(pd.read_csv(self.path / "out" / "role_attention.csv").empty)
        self.assertTrue(pd.read_csv(self.path / "out" / "depth_attention.csv").empty)
        self.assertTrue(pd.read_csv(self.path / "out" / "attention_overlap.csv").empty)
        self.assertTrue(pd.read_csv(self.path / "out" / "route_nodes.csv").empty)
        self.assertTrue(pd.read_csv(self.path / "out" / "cycle_nodes.csv").empty)
        self.assertEqual(pd.read_csv(self.path / "out" / "depth_summary.csv").n_nodes.sum(), 2)
        self.assertEqual(pd.read_csv(self.path / "out" / "cluster_depths.csv").n_nodes.sum(), 2)
        self.assertEqual(pd.read_csv(self.path / "out" / "role_depths.csv").n_nodes.sum(), 2)
        self.assertTrue(pd.read_csv(self.path / "out" / "role_flows.csv").empty)
        self.assertTrue(pd.read_csv(self.path / "out" / "depth_flows.csv").empty)
        self.assertTrue(pd.read_csv(self.path / "out" / "top_counterparties.csv").empty)
        self.assertTrue((resilience.edges_left == 0).all())
        self.assertTrue((resilience.seed_reach_share == 0).all())
        self.assertEqual(len(pd.read_csv(self.path / "out" / "data_gaps.csv")), 6)
        self.assertTrue((self.path / "out" / "network.html").is_file())


if __name__ == "__main__":
    unittest.main()
