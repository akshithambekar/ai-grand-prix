import unittest

import numpy as np

from controls.reward import RewardCalculator


def data(gate_index=0, track_id=1, collision=None):
    value = {
        "race_status": {"active_gate_index": gate_index},
        "gate": {"detected": True, "track_id": track_id},
    }
    if collision is not None:
        value["collision"] = collision
    return value


def observation(cx=0.2, cy=0.1, log_area=0.0, detected=True):
    value = np.zeros(19, dtype=np.float32)
    value[0] = float(detected)
    value[1:4] = (cx, cy, log_area)
    return value


class RewardCalculatorTests(unittest.TestCase):
    def test_centering_progress_is_positive(self):
        rewards = RewardCalculator()
        rewards.reset(data(), observation(cx=0.4, cy=0.0))
        result = rewards.compute(data(), observation(cx=0.2, cy=0.0))
        self.assertGreater(result.components["visual_progress"], 0.0)

    def test_gate_increment_is_rewarded_once(self):
        rewards = RewardCalculator()
        rewards.reset(data(), observation())
        first = rewards.compute(data(gate_index=1, track_id=2), observation(), None)
        second = rewards.compute(data(gate_index=1, track_id=2), observation(), None)
        self.assertEqual(first.components["gate_pass"], 20.0)
        self.assertNotIn("gate_pass", second.components)

    def test_target_change_has_no_dense_spike(self):
        rewards = RewardCalculator()
        rewards.reset(data(), observation(cx=0.0, log_area=1.0))
        result = rewards.compute(
            data(track_id=2),
            observation(cx=0.9, log_area=-1.0),
        )
        self.assertNotIn("visual_progress", result.components)

    def test_collision_is_deduplicated(self):
        collision = {"sequence": 1, "collision_id": 1002, "threat_level": 2}
        rewards = RewardCalculator()
        rewards.reset(data(), observation())
        first = rewards.compute(data(collision=collision), observation())
        second = rewards.compute(data(collision=collision), observation())
        self.assertEqual(first.components["collision"], -20.0)
        self.assertNotIn("collision", second.components)


if __name__ == "__main__":
    unittest.main()
