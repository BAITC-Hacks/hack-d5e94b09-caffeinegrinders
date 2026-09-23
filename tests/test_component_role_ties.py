"""Regression checks for component role ties."""

import unittest
import pandas as pd
import networkx as nx
from run import component_role_matrix


class RegressionTest(unittest.TestCase):

    def test_equal_size_components_are_ordered_by_gid_and_roles_use_local_shares(self):
        graph = nx.DiGraph([(4, 3), (2, 1)])
        df = pd.DataFrame({
            "gid": [4, 3, 2, 1], "role": ["terminal", "terminal", "transit", "distributor"],
            "is_seed": [False, False, False, True],
            "boundary": [True, False, False, False], "priority_score": [.5, .5, .6, .9],
        })
        result = component_role_matrix(graph, df).set_index(["component_id", "role"])
        self.assertEqual(set(result.index), {(1, "distributor"), (1, "transit"), (2, "terminal")})
        self.assertEqual(result.loc[(1, "transit")].share_component, .5)
        self.assertEqual(result.loc[(1, "distributor")].n_seed, 1)
        row = result.loc[(2, "terminal")]
        self.assertEqual(row.n_nodes, 2)
        self.assertEqual(row.share_component, 1.0)
        self.assertEqual(row.top_gids, "3;4")
        self.assertEqual(row.boundary_nodes, 1)
        self.assertEqual(row.avg_priority_score, .5)


if __name__ == "__main__":
    unittest.main()

