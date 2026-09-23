"""Return-flow detection must respect its four-transfer observation limit."""

import unittest

import networkx as nx
import pandas as pd

from run import find_cycles


class CycleLimitsTest(unittest.TestCase):
    def test_four_transfer_cycle_is_included_and_five_transfer_cycle_excluded(self):
        graph = nx.DiGraph()
        # Two disconnected rings ensure that there are no shorter alternative cycles.
        for src, dst, amount in [
            (1, 2, 100.0), (2, 3, 80.0), (3, 4, 60.0), (4, 1, 40.0),
            (5, 6, 100.0), (6, 7, 100.0), (7, 8, 100.0),
            (8, 9, 100.0), (9, 5, 100.0),
        ]:
            graph.add_edge(src, dst, sum_kzt=amount)
        nodes = pd.DataFrame({"gid": list(range(1, 10)),
                              "is_seed": [gid in (1, 5) for gid in range(1, 10)]})
        features, cycles = find_cycles(graph, nodes)
        self.assertEqual(len(cycles), 1)
        cycle = cycles.iloc[0]
        self.assertEqual(cycle.length, 4)
        self.assertEqual(cycle.path, "1 → 2 → 3 → 4 → 1")
        self.assertEqual(cycle.bottleneck_kzt, 40.0)
        self.assertEqual(cycle.n_seed, 1)
        features = features.set_index("gid")
        self.assertEqual(features.cycles.tolist(), [1, 1, 1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(features.cycle_with_seed.tolist(),
                         [True, True, True, True, False, False, False, False, False])


if __name__ == "__main__":
    unittest.main()
