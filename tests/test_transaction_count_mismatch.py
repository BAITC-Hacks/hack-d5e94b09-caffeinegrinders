"""Regression checks for transaction count mismatch."""

import unittest
import pandas as pd
import tempfile
from pathlib import Path
from run import load_data


class RegressionTest(unittest.TestCase):

    def test_matching_amounts_do_not_hide_wrong_transaction_count(self):
        nodes = pd.DataFrame({"gid": [1, 2], "depth": [0, 1], "is_seed": [True, False]})
        edges = pd.DataFrame({"src": [1], "dst": [2], "sum_kzt": [10000.0], "n_tx": [2]})
        tx = pd.DataFrame({"src": [1], "dst": [2], "date": ["2026-07-01"], "sum_kzt": [10000.0]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for name, frame in (("nodes", nodes), ("edges", edges), ("transactions", tx)):
                frame.to_parquet(path / f"{name}.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "transactions and edges disagree"):
                load_data(path)
            edges["n_tx"] = [1]
            edges.to_parquet(path / "edges.parquet", index=False)
            _, valid_edges, valid_tx = load_data(path)
            self.assertEqual(valid_edges.n_tx.sum(), len(valid_tx))


if __name__ == "__main__":
    unittest.main()

