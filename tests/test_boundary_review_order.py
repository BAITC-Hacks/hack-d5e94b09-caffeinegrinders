"""Regression coverage for boundary review order."""

import unittest

import pandas as pd

from run import boundary_review


class SummaryRegressionTest(unittest.TestCase):

    def test_boundary_queue_uses_continuation_amount_degree_then_gid(self):
        df = pd.DataFrame({
            "gid": [5, 4, 3, 2, 1, 99],
            "boundary": [True] * 5 + [False],
            "p_hidden_outgoing": [.5, .8, .8, .8, .8, 1.0],
            "in_kzt": [1000, 100, 200, 200, 200, 9999],
            "in_deg": [10, 1, 1, 2, 2, 99],
            "depth": [4] * 5 + [1], "cluster_id": [1] * 6,
            "priority_score": [.9, .8, .7, .6, .5, 1.0],
            "attention": ["нет"] * 6,
        })
        result = boundary_review(df)
        self.assertEqual(result.gid.tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(result["rank"].tolist(), [1, 2, 3, 4, 5])
        for row in result.itertuples():
            self.assertIn(f"gid {row.gid}", row.next_request)
        self.assertEqual(len(df), 6)
        self.assertNotIn("rank", df.columns)


if __name__ == "__main__":
    unittest.main()

