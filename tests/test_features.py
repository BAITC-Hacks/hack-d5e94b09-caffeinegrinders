"""Check graph signals against small transaction histories with known outcomes."""

import unittest

import pandas as pd

from run import daily_timeline, graph_features


class GraphFeaturesTest(unittest.TestCase):
    def test_rapid_48h_respects_time_window_and_counts_each_incoming_once(self):
        nodes = pd.DataFrame({
            "gid": [1, 2, 3], "depth": [0, 1, 2],
            "is_seed": [True, False, False],
        })
        received = pd.Timestamp("2026-07-10 12:00:00")
        cases = [
            ("before_receipt", [-1], 0.0),
            ("same_time", [0], 1.0),
            ("exactly_48_hours", [172800], 1.0),
            ("after_48_hours", [172801], 0.0),
            ("no_outgoing", [], 0.0),
            ("multiple_outgoing_for_one_of_two_receipts", [3600, 7200], 0.5),
        ]
        for name, offsets, expected in cases:
            with self.subTest(case=name):
                rows = [{"src": 1, "dst": 2, "date": received, "sum_kzt": 10000.0}]
                if name == "multiple_outgoing_for_one_of_two_receipts":
                    rows.append({"src": 1, "dst": 2,
                                 "date": received + pd.Timedelta(days=4),
                                 "sum_kzt": 10000.0})
                rows.extend({"src": 2, "dst": 3,
                             "date": received + pd.Timedelta(seconds=offset),
                             "sum_kzt": 5000.0} for offset in offsets)
                tx = pd.DataFrame(rows)
                edges = tx.groupby(["src", "dst"], as_index=False).agg(
                    sum_kzt=("sum_kzt", "sum"), n_tx=("sum_kzt", "size"),
                )
                _, features = graph_features(nodes, edges, tx)
                rapid = features.set_index("gid").rapid_48h
                self.assertAlmostEqual(rapid.loc[2], expected)
                self.assertEqual(rapid.loc[1], 0.0)  # No incoming transfers.
                self.assertEqual(rapid.loc[3], 0.0)  # No outgoing transfers.

    def test_daily_timeline_splits_incoming_and_outgoing_activity(self):
        tx = pd.DataFrame([
            {"src": 1, "dst": 2, "date": pd.Timestamp("2026-07-01 09:00"), "sum_kzt": 100.0},
            {"src": 3, "dst": 2, "date": pd.Timestamp("2026-07-01 10:00"), "sum_kzt": 200.0},
            {"src": 2, "dst": 4, "date": pd.Timestamp("2026-07-01 11:00"), "sum_kzt": 50.0},
            {"src": 2, "dst": 5, "date": pd.Timestamp("2026-07-02 11:00"), "sum_kzt": 75.0},
        ])
        rows = daily_timeline(tx).set_index(["gid", "date"])
        first_day = rows.loc[(2, "2026-07-01")]
        self.assertEqual(first_day.in_tx, 2)
        self.assertEqual(first_day.out_tx, 1)
        self.assertEqual(first_day.in_kzt, 300.0)
        self.assertEqual(first_day.out_kzt, 50.0)
        self.assertEqual(first_day.net_kzt, 250.0)
        self.assertEqual(first_day.unique_payers, 2)
        self.assertEqual(first_day.unique_recipients, 1)
        self.assertEqual(rows.loc[(2, "2026-07-02")].net_kzt, -75.0)


if __name__ == "__main__":
    unittest.main()
