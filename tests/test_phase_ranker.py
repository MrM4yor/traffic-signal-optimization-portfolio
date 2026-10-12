import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
"""Offline action-grouping and route-holdout invariants; no model fit or SUMO."""

import unittest
import pandas as pd
from train_neural_phase_ranker import decision_metrics, split_routes


class RouteSplitTests(unittest.TestCase):
    def test_all_branches_stay_together_and_splits_are_disjoint(self):
        rows = pd.DataFrame(
            [
                {
                    "route": f"r{route:02}",
                    "state_time_s": 1,
                    "tls_id": "C",
                    "regret": regret,
                }
                for route in range(15)
                for regret in [0.0, 2.0]
            ]
        )
        groups = split_routes(rows.sample(frac=1, random_state=1))
        route_sets = [set(group.route) for group in groups.values()]
        self.assertEqual(sum(map(len, route_sets)), 15)
        for i, left in enumerate(route_sets):
            for right in route_sets[i + 1 :]:
                self.assertFalse(left & right)
        self.assertTrue(
            all((group.groupby("route").size() == 2).all() for group in groups.values())
        )

    def test_insufficient_routes_rejected(self):
        with self.assertRaises(ValueError):
            split_routes(pd.DataFrame({"route": ["a", "b", "c", "d"]}))

    def test_action_regret_uses_each_decision_not_global_minimum(self):
        rows = pd.DataFrame(
            {
                "route": ["a"] * 4,
                "state_time_s": [1, 1, 2, 2],
                "tls_id": ["C"] * 4,
                "regret": [0.0, 2.0, 3.0, 0.0],
            }
        )
        result = decision_metrics(rows, [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(result["mean_selected_regret"], 1.5)
        self.assertEqual(result["exact_best_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
