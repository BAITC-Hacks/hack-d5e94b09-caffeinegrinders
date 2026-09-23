"""Regression checks for duplicate aggregated edges."""

import unittest
import pandas as pd
import tempfile
from pathlib import Path
from run import load_data


class RegressionTest(unittest.TestCase):

    def test_duplicate_edge_pairs_are_rejected_even_when_totals_would_match(self):
        nodes = pd.DataFrame({"gid": [1, 2], "depth": [0, 1], "is_seed": [True, False]})
        edges = pd.DataFrame({"src": [1, 1], "dst": [2, 2],
                              "sum_kzt": [5000.0, 5000.0], "n_tx": [1, 1]})
        tx = pd.DataFrame({"src": [1, 1], "dst": [2, 2],
                           "date": ["2026-07-01", "2026-07-02"], "sum_kzt": [5000.0, 5000.0]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for name, frame in (("nodes", nodes), ("edges", edges), ("transactions", tx)):
                frame.to_parquet(path / f"{name}.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate aggregated edges"):
                load_data(path)


if __name__ == "__main__":
    unittest.main()

