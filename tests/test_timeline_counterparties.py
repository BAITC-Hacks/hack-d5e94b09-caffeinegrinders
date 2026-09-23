"""Daily counterparties must count people, not repeated transfers."""

import unittest

import pandas as pd

from run import daily_timeline


class TimelineCounterpartiesTest(unittest.TestCase):
    def test_repeated_transfers_count_once_per_counterparty_per_day(self):
        tx = pd.DataFrame([
            (1, 2, "2026-07-01 09:00", 100.0),
            (1, 2, "2026-07-01 10:00", 200.0),
            (3, 2, "2026-07-01 11:00", 300.0),
            (2, 4, "2026-07-01 12:00", 50.0),
            (2, 4, "2026-07-01 13:00", 75.0),
            (1, 2, "2026-07-02 09:00", 400.0),
        ], columns=["src", "dst", "date", "sum_kzt"])
        tx["date"] = pd.to_datetime(tx.date)
        timeline = daily_timeline(tx).set_index(["gid", "date"])
        first = timeline.loc[(2, "2026-07-01")]
        self.assertEqual(first.in_tx, 3)
        self.assertEqual(first.unique_payers, 2)
        self.assertEqual(first.out_tx, 2)
        self.assertEqual(first.unique_recipients, 1)
        self.assertEqual(first.in_kzt, 600.0)
        self.assertEqual(first.out_kzt, 125.0)
        self.assertEqual(first.net_kzt, 475.0)
        second = timeline.loc[(2, "2026-07-02")]
        self.assertEqual(second.in_tx, 1)
        self.assertEqual(second.unique_payers, 1)
        self.assertEqual(second.out_tx, 0)
        self.assertEqual(second.unique_recipients, 0)
        self.assertEqual(second.net_kzt, 400.0)
        self.assertTrue(timeline.index.is_unique)


if __name__ == "__main__":
    unittest.main()
