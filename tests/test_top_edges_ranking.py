"""The largest-transfer shortlist must be stable when amounts tie."""

import unittest

import pandas as pd

from run import top_edge_summary


class TopEdgesRankingTest(unittest.TestCase):
    def test_tied_amounts_have_stable_order_before_limit_is_applied(self):
        nodes = pd.DataFrame({
            "gid": [1, 2, 3, 4], "role": ["peripheral"] * 4,
            "cluster_id": [1] * 4, "is_seed": [False] * 4,
        })
        edges = pd.DataFrame([
            (3, 4, 500.0, 1), (1, 3, 500.0, 2),
            (1, 2, 500.0, 3), (4, 1, 900.0, 4), (2, 4, 100.0, 5),
        ], columns=["src", "dst", "sum_kzt", "n_tx"])
        for seed in (7, 42, 2026):
            with self.subTest(shuffle=seed):
                result = top_edge_summary(edges.sample(frac=1, random_state=seed), nodes, limit=3)
                self.assertEqual(list(zip(result.src, result.dst)), [(4, 1), (1, 2), (1, 3)])
                self.assertEqual(result["rank"].tolist(), [1, 2, 3])
                self.assertEqual(result.sum_kzt.tolist(), [900.0, 500.0, 500.0])
                self.assertEqual(result.n_tx.tolist(), [4, 3, 2])


if __name__ == "__main__":
    unittest.main()
