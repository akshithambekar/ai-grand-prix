import unittest

import numpy as np

from controls.reward import RewardCalculator


def data(gate_index=0, track_id=1, collision=None, active_gate=None):
    value = {
        "race_status": {"active_gate_index": gate_index},
        "gate": {"detected": True, "track_id": track_id},
    }
    if collision is not None:
        value["collision"] = collision
    if active_gate is not None:
        value["active_gate_state"] = active_gate
    return value


def physical_state(*, down=0.0, down_velocity=0.0, roll=0.0, pitch=0.0):
    return {
        "valid": True,
        "position_ned": (0.0, 0.0, down),
        "velocity_ned": (0.0, 0.0, down_velocity),
        "euler": (roll, pitch, 0.0),
    }


def observation(cx=0.2, cy=0.1, log_area=0.0, detected=True, action=None):
    value = np.zeros(38, dtype=np.float32)
    value[0] = float(detected)
    value[1:4] = (cx, cy, log_area)
    if action is not None:
        value[-4:] = action
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

    def test_physical_progress_replaces_visual_progress(self):
        previous = {
            "valid": True, "gate_id": 0, "distance_m": 10.0,
            "lateral_m": 0.2, "vertical_m": 0.1, "width_m": 2.0, "height_m": 2.0,
        }
        current = dict(previous, distance_m=9.7, lateral_m=0.1)
        rewards = RewardCalculator()
        rewards.reset(data(active_gate=previous), observation(cx=0.4))
        result = rewards.compute(data(active_gate=current), observation(cx=0.2))
        self.assertGreater(result.components["position_progress"], 0.0)
        self.assertLess(result.components["position_progress"], 0.3)
        self.assertIn("gate_alignment", result.components)
        self.assertNotIn("visual_progress", result.components)

    def test_forward_progress_is_suppressed_while_misaligned(self):
        aligned = {
            "valid": True, "gate_id": 0, "distance_m": 10.0,
            "lateral_m": 0.0, "vertical_m": 0.0, "width_m": 2.0, "height_m": 2.0,
        }
        misaligned = dict(aligned, lateral_m=1.0)

        aligned_rewards = RewardCalculator()
        aligned_rewards.reset(data(active_gate=aligned), observation())
        aligned_result = aligned_rewards.compute(
            data(active_gate=dict(aligned, distance_m=9.8)), observation()
        )

        misaligned_rewards = RewardCalculator()
        misaligned_rewards.reset(data(active_gate=misaligned), observation())
        misaligned_result = misaligned_rewards.compute(
            data(active_gate=dict(misaligned, distance_m=9.8)), observation()
        )

        self.assertGreater(aligned_result.components["position_progress"], 0.19)
        self.assertLess(misaligned_result.components["position_progress"], 0.02)
        self.assertEqual(misaligned_result.components["gate_alignment"], -0.2)

    def test_improving_alignment_is_rewarded(self):
        previous = {
            "valid": True, "gate_id": 0, "distance_m": 10.0,
            "lateral_m": 0.8, "vertical_m": 0.0, "width_m": 2.0, "height_m": 2.0,
        }
        current = dict(previous, lateral_m=0.4)
        rewards = RewardCalculator()
        rewards.reset(data(active_gate=previous), observation())
        result = rewards.compute(data(active_gate=current), observation())
        self.assertAlmostEqual(result.components["alignment_progress"], 0.1)
        self.assertGreater(result.components["alignment_progress"], 0.0)

    def test_descent_altitude_loss_and_tilt_are_penalized_without_speed_penalty(self):
        value = data()
        value["vehicle_state"] = physical_state()
        rewards = RewardCalculator()
        rewards.reset(value, observation())

        moving = data()
        moving["vehicle_state"] = physical_state(
            down=1.25, down_velocity=1.25, roll=0.5, pitch=0.5,
        )
        result = rewards.compute(moving, observation())

        self.assertAlmostEqual(result.components["descent_stability"], -0.05)
        self.assertAlmostEqual(result.components["altitude_stability"], -0.05)
        self.assertAlmostEqual(result.components["attitude_stability"], -0.1)
        self.assertNotIn("misaligned_speed", result.components)
        self.assertNotIn("speed", result.components)

    def test_level_horizontal_speed_has_no_vehicle_stability_penalty(self):
        value = data()
        value["vehicle_state"] = physical_state()
        rewards = RewardCalculator()
        rewards.reset(value, observation())
        value["vehicle_state"] = dict(physical_state(), velocity_ned=(15.0, -10.0, 0.0))
        result = rewards.compute(value, observation())
        self.assertEqual(result.components["descent_stability"], 0.0)
        self.assertEqual(result.components["altitude_stability"], 0.0)
        self.assertEqual(result.components["attitude_stability"], 0.0)

    def test_terminal_failures_receive_explicit_penalties(self):
        for reason, expected in (("stuck", -20.0), ("gate_timeout", -10.0)):
            rewards = RewardCalculator()
            rewards.reset(data(), observation())
            result = rewards.compute(data(), observation(), reason)
            self.assertEqual(result.components["terminal_failure"], expected)

    def test_invalid_physical_state_uses_visual_fallback(self):
        rewards = RewardCalculator()
        rewards.reset(data(active_gate={"valid": False}), observation(cx=0.4))
        result = rewards.compute(data(active_gate={"valid": False}), observation(cx=0.2))
        self.assertGreater(result.components["visual_progress"], 0.0)

    def test_unchanged_actions_have_no_slew_penalty(self):
        rewards = RewardCalculator()
        command = [0.2, -0.1, 0.3, 0.4]
        rewards.reset(data(), observation(action=command))
        result = rewards.compute(data(), observation(action=command))
        self.assertEqual(result.components["rate_slew"], 0.0)
        self.assertEqual(result.components["thrust_slew"], 0.0)

    def test_all_action_changes_receive_bounded_slew_penalties(self):
        rewards = RewardCalculator()
        rewards.reset(data(), observation(action=[-1.0, -1.0, -1.0, -1.0]))
        result = rewards.compute(data(), observation(action=[1.0, 1.0, 1.0, 1.0]))
        self.assertAlmostEqual(result.components["rate_slew"], -0.24)
        self.assertAlmostEqual(result.components["thrust_slew"], -0.02)
        self.assertAlmostEqual(
            result.components["step"]
            + result.components["rate_slew"]
            + result.components["thrust_slew"],
            -0.27,
        )

    def test_slew_state_is_cleared_on_episode_reset(self):
        rewards = RewardCalculator()
        rewards.reset(data(), observation(action=[1.0, 0.0, 0.0, 0.0]))
        rewards.compute(data(), observation(action=[-1.0, 0.0, 0.0, 0.0]))
        rewards.reset(data(), observation(action=[0.0, 0.0, 0.0, 0.0]))
        result = rewards.compute(data(), observation(action=[0.0, 0.0, 0.0, 0.0]))
        self.assertEqual(result.components["rate_slew"], 0.0)


if __name__ == "__main__":
    unittest.main()
