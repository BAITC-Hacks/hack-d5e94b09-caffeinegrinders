"""Transfer direction must be preserved when attaching endpoint metadata."""

import unittest

import pandas as pd

from run import top_edge_summary


class TopEdgesMetadataTest(unittest.TestCase):
    def test_reversed_edges_keep_sender_and_recipient_metadata(self):
        # Deliberately unordered rows prevent positional joins from passing.
        nodes = pd.DataFrame([
            (30, "terminal", 7, False),
            (10, "distributor", 7, True),
            (20, "consolidator", 9, False),
        ], columns=["gid", "role", "cluster_id", "is_seed"])
        edges = pd.DataFrame([
            (10, 20, 900.0, 2), (20, 10, 600.0, 3), (10, 30, 300.0, 1),
        ], columns=["src", "dst", "sum_kzt", "n_tx"])
        result = top_edge_summary(edges, nodes).set_index(["src", "dst"])
        expected = {
            (10, 20): ("distributor", "consolidator", 7, 9, True, False, False),
            (20, 10): ("consolidator", "distributor", 9, 7, False, True, False),
            (10, 30): ("distributor", "terminal", 7, 7, True, False, True),
        }
        columns = ["src_role", "dst_role", "src_cluster", "dst_cluster",
                   "src_seed", "dst_seed", "same_cluster"]
        self.assertEqual(set(result.index), set(expected))
        for endpoints, metadata in expected.items():
            with self.subTest(edge=endpoints):
                self.assertEqual(tuple(result.loc[endpoints, columns]), metadata)


if __name__ == "__main__":
    unittest.main()
