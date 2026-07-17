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
    }


class ObservationEncoderTests(unittest.TestCase):
    def test_centered_gate_and_imu_are_normalized(self):
        observation = ObservationEncoder().encode(snapshot(), np.zeros(4))
        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertAlmostEqual(observation[1], 0.0)
        self.assertAlmostEqual(observation[2], 0.0)
        self.assertAlmostEqual(observation[9], 0.2)
        self.assertTrue(np.isfinite(observation).all())

    def test_missing_data_produces_finite_vector(self):
        observation = ObservationEncoder().encode({}, [0, 0, 0, 0])
        self.assertEqual(observation[0], 0.0)
        self.assertTrue(np.isfinite(observation).all())

    def test_deltas_require_same_track_and_gate(self):
        encoder = ObservationEncoder()
        encoder.encode(snapshot(), np.zeros(4))
        moved = encoder.encode(snapshot(cx=330, cy=170, area=12_000), np.zeros(4))
        self.assertNotEqual(tuple(moved[4:7]), (0.0, 0.0, 0.0))

        switched = encoder.encode(
            snapshot(cx=500, cy=100, area=500, track_id=2, gate_index=1),
            np.zeros(4),
        )
        self.assertEqual(tuple(switched[4:7]), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
