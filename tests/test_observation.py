import unittest

import numpy as np

from controls.observation import OBSERVATION_SIZE, ObservationEncoder


def snapshot(cx=320, cy=180, area=10_000, detected=True, track_id=1, gate_index=0):
    return {
        "gate": {
            "detected": detected,
            "centroid": (cx, cy) if detected else None,
            "frame_size": (640, 360),
            "area_px": area if detected else None,
            "track_id": track_id,
            "tracking_confidence": 0.75,
            "tracking_missed_frames": 0,
        },
        "race_status": {"active_gate_index": gate_index},
        "imu": {
            "xgyro": 1.0, "ygyro": -1.0, "zgyro": 0.0,
            "xacc": 2.0, "yacc": -2.0, "zacc": 9.8,
        },
        "vehicle_state": {
            "valid": True,
            "position_ned": (10.0, -20.0, 5.0),
            "velocity_ned": (2.0, -4.0, 1.0),
            "euler": (0.0, 0.0, 0.0),
            "body_rates": (1.0, -1.0, 0.0),
            "acceleration_body": (2.0, -2.0, 9.8),
        },
        "active_gate_state": {
            "valid": True,
            "relative_position_body": (25.0, -5.0, 2.5),
            "relative_position_gate": (25.0, -5.0, 2.5),
            "gate_normal_body": (-1.0, 0.0, 0.0),
            "width_m": 2.0,
            "height_m": 2.0,
        },
    }


class ObservationEncoderTests(unittest.TestCase):
    def test_centered_gate_and_imu_are_normalized(self):
        observation = ObservationEncoder().encode(snapshot(), np.zeros(4))
        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertAlmostEqual(observation[1], 0.0)
        self.assertAlmostEqual(observation[2], 0.0)
        self.assertEqual(observation.shape, (38,))
        self.assertAlmostEqual(observation[7], 0.1)
        self.assertAlmostEqual(observation[19], 0.2)
        self.assertAlmostEqual(observation[22], 0.1)
        self.assertAlmostEqual(observation[26], 0.5)
        self.assertAlmostEqual(observation[32], 0.2)
        self.assertTrue(np.isfinite(observation).all())

    def test_missing_data_produces_finite_vector(self):
        observation = ObservationEncoder().encode({}, [0, 0, 0, 0])
        self.assertEqual(observation[0], 0.0)
        self.assertTrue(np.isfinite(observation).all())

    def test_invalid_state_groups_are_zero_with_masks(self):
        value = snapshot()
        value["vehicle_state"] = {"valid": False}
        value["active_gate_state"] = {"valid": False}
        observation = ObservationEncoder().encode(value, np.ones(4))
        self.assertEqual(observation[6], 0.0)
        self.assertEqual(tuple(observation[7:25]), (0.0,) * 18)
        self.assertEqual(observation[25], 0.0)
        self.assertEqual(tuple(observation[26:34]), (0.0,) * 8)
        self.assertEqual(tuple(observation[34:38]), (1.0,) * 4)


if __name__ == "__main__":
    unittest.main()
