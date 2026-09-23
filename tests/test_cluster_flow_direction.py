"""Cross-cluster totals must preserve direction and exclude internal transfers."""

import unittest

import pandas as pd

from run import cluster_flows


class ClusterFlowDirectionTest(unittest.TestCase):
    def test_cross_cluster_flows_are_aggregated_separately_by_direction(self):
        nodes = pd.DataFrame({"gid": [1, 2, 3, 4], "cluster_id": [10, 10, 20, 20]})
        edges = pd.DataFrame([
            (1, 3, 100.0, 2), (2, 4, 200.0, 3),
            (3, 1, 75.0, 4),
            (1, 2, 10000.0, 50), (3, 4, 20000.0, 60),
        ], columns=["src", "dst", "sum_kzt", "n_tx"])
        flows = cluster_flows(nodes, edges)
        self.assertEqual(flows.to_dict("records"), [
            {"src_cluster": 10, "dst_cluster": 20, "sum_kzt": 300.0,
             "n_edges": 2, "n_tx": 5},
            {"src_cluster": 20, "dst_cluster": 10, "sum_kzt": 75.0,
             "n_edges": 1, "n_tx": 4},
        ])


if __name__ == "__main__":
    unittest.main()
