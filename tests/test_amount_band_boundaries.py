"""Regression coverage for amount band boundaries."""

import unittest

import pandas as pd

from run import amount_band_summary


class SummaryRegressionTest(unittest.TestCase):

    def test_exact_band_boundaries_belong_to_next_band_once(self):
        tx = pd.DataFrame({"sum_kzt": [
            5000.0, 9999.99, 10000.0, 49999.99, 50000.0,
            99999.99, 100000.0, 499999.99, 500000.0,
        ]})
        result = amount_band_summary(tx)
        self.assertEqual(result.amount_band.tolist(),
                         ["5k-10k", "10k-50k", "50k-100k", "100k-500k", "500k+"])
        self.assertEqual(result.n_tx.tolist(), [2, 2, 2, 2, 1])
        self.assertEqual(result.sum_kzt.tolist(),
                         [14999.99, 59999.99, 149999.99, 599999.99, 500000.0])
        self.assertEqual(result.n_tx.sum(), 9)
        self.assertAlmostEqual(result.sum_kzt.sum(), tx.sum_kzt.sum(), places=2)
        self.assertEqual(result.share_tx.tolist(), [.2222, .2222, .2222, .2222, .1111])


if __name__ == "__main__":
    unittest.main()

