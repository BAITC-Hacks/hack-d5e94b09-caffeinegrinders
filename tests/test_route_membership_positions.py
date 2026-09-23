"""Regression coverage for route membership positions."""

import unittest

import pandas as pd

from run import route_node_membership


class SummaryRegressionTest(unittest.TestCase):

    def test_membership_preserves_route_order_and_large_identifiers(self):
        a, b, c = 9007199254740993, 9007199254740995, 9007199254740997
        df = pd.DataFrame({
            "gid": [c, a, b], "role": ["terminal", "distributor", "transit"],
            "cluster_id": [3, 1, 2], "is_seed": [False, True, False],
            "priority_score": [.3, .1, .2],
        })
        routes = pd.DataFrame([
            {"route_id": 8, "kind": "chain", "path": f"{a} → {b} → {c}"},
            {"route_id": 9, "kind": "chain", "path": f"{b}→{c}"},
        ])
        result = route_node_membership(routes, df)
        self.assertEqual(result.gid.tolist(), [a, b, c, b, c])
        self.assertEqual(result.position.tolist(), [1, 2, 3, 1, 2])
        self.assertEqual(result.route_id.tolist(), [8, 8, 8, 9, 9])
        self.assertEqual(result.role.tolist(),
                         ["distributor", "transit", "terminal", "transit", "terminal"])
        self.assertEqual(result.cluster_id.tolist(), [1, 2, 3, 2, 3])
        self.assertEqual(result.is_seed.tolist(), [True, False, False, False, False])
        self.assertEqual(result.priority_score.tolist(), [.1, .2, .3, .2, .3])


if __name__ == "__main__":
    unittest.main()

