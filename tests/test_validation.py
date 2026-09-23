"""Regression checks for data that previously passed validation silently."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from run import load_data, graph_features, assign_clusters, score_roles, rank_nodes, write_outputs


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
        df = rank_nodes(score_roles(assign_clusters(graph, df, edges)))
        write_outputs(df, edges, self.path / "out")
        self.assertEqual(df.cluster_id.nunique(), 2)
        self.assertEqual(len(pd.read_csv(self.path / "out" / "top_nodes.csv")), 2)


if __name__ == "__main__":
    unittest.main()
