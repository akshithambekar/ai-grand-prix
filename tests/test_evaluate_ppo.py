import unittest

import numpy as np

from scripts.evaluate_ppo import _trace_row


class EvaluationTraceTests(unittest.TestCase):
    def test_trace_row_contains_gate_plane_state_and_reward_components(self):
        info = {
            "detected": False,
            "termination_reason": None,
            "vehicle_state": {
                "position_ned": (1.0, 2.0, 3.0),
                "velocity_ned": (4.0, 5.0, 6.0),
                "euler": (0.1, 0.2, 0.3),
            },
            "active_gate_state": {
                "valid": True,
                "plane_distance_m": 2.0,
                "signed_plane_distance_m": -2.0,
                "lateral_m": 0.5,
                "vertical_m": 0.25,
                "width_m": 2.0,
                "height_m": 2.0,
            },
            "reward_components": {
                "aligned_approach": 0.1,
                "alignment_progress": 0.2,
            },
        }

        row = _trace_row(2, 3, np.zeros(4), 0.3, info)

        self.assertEqual(row["gate_phase"], "commit")
        self.assertAlmostEqual(row["alignment_error"], 0.559016994)
        self.assertEqual(row["aligned_approach"], 0.1)
        self.assertEqual(row["alignment_progress"], 0.2)


if __name__ == "__main__":
    unittest.main()
