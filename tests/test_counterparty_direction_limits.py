"""Regression coverage for counterparty direction limits."""

import unittest

import pandas as pd

from run import top_counterparties


class SummaryRegressionTest(unittest.TestCase):

    def test_limit_applies_per_node_and_direction_with_stable_ties(self):
        df = pd.DataFrame({
            "gid": [1, 2, 3, 4, 5], "role": ["transit"] * 5,
            "cluster_id": [1, 2, 3, 4, 5], "is_seed": [False, True, False, False, False],
        })
        edges = pd.DataFrame([
            (1, 3, 100.0, 2), (1, 2, 100.0, 3), (1, 4, 50.0, 1),
            (5, 1, 300.0, 4), (2, 1, 200.0, 5), (3, 1, 20.0, 1),
        ], columns=["src", "dst", "sum_kzt", "n_tx"])
        result = top_counterparties(edges, df, per_node=2)
        owner = result[result.gid == 1]
        incoming = owner[owner.direction == "in"]
        outgoing = owner[owner.direction == "out"]
        self.assertEqual(incoming.counterparty_gid.tolist(), [5, 2])
        self.assertEqual(outgoing.counterparty_gid.tolist(), [2, 3])
        self.assertEqual(incoming["rank"].tolist(), [1, 2])
        self.assertEqual(outgoing["rank"].tolist(), [1, 2])
        self.assertEqual(outgoing.n_tx.tolist(), [3, 2])
        self.assertEqual(outgoing.counterparty_cluster.tolist(), [2, 3])
        self.assertEqual(outgoing.counterparty_seed.tolist(), [True, False])
        self.assertTrue((result.groupby(["gid", "direction"]).size() <= 2).all())
        # Other accounts must retain their own lists; the limit is not global.
        self.assertEqual(set(result.gid), {1, 2, 3, 4, 5})


if __name__ == "__main__":
    unittest.main()

