"""Regression coverage for daily active nodes."""

import unittest

import pandas as pd

from run import daily_summary


class SummaryRegressionTest(unittest.TestCase):

    def test_daily_activity_counts_union_of_senders_and_recipients(self):
        tx = pd.DataFrame([
            (2, 3, "2026-07-02 01:00", 40.0),
            (1, 2, "2026-07-01 09:00", 10.0),
            (2, 1, "2026-07-01 10:00", 20.0),
            (1, 2, "2026-07-01 11:00", 30.0),
        ], columns=["src", "dst", "date", "sum_kzt"])
        tx["date"] = pd.to_datetime(tx.date)
        result = daily_summary(tx)
        self.assertEqual(result.to_dict("records"), [
            {"date": "2026-07-01", "n_tx": 3, "sum_kzt": 60.0,
             "unique_senders": 2, "unique_recipients": 2, "active_nodes": 2},
            {"date": "2026-07-02", "n_tx": 1, "sum_kzt": 40.0,
             "unique_senders": 1, "unique_recipients": 1, "active_nodes": 2},
        ])


if __name__ == "__main__":
    unittest.main()

