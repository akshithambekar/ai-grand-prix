import unittest

import numpy as np
from stable_baselines3.common.env_checker import check_env

from controls.episode_manager import EpisodeConfig, EpisodeManager
from gym_env.ai_gp_env import AIGPEnv


class FakeClock:
    def __init__(self, data):
        self.now = 0.0
        self.data = data
        self.reset_at = None
        self.next_start_ms = 5000
        self.next_frame_id = 1

    def __call__(self):
        return self.now

    def request_reset(self):
        self.reset_at = self.now
        self.next_frame_id += 1
        current = self.data.get("race_status", {}).get("race_start_boot_time_ms", 4000)
        self.next_start_ms = current + 1000

    def sleep(self, duration):
        self.now += max(0.0, duration)
        if self.reset_at is None:
            return
        elapsed = self.now - self.reset_at
        sim_ms = self.next_start_ms if elapsed >= 4.0 else self.next_start_ms - 3000 + int(elapsed * 1000)
        self.data["race_status"] = {
            "sim_boot_time_ms": sim_ms,
            "race_start_boot_time_ms": self.next_start_ms,
            "race_finish_time_ns": -1,
            "active_gate_index": 0,
            "received_at_s": self.now,
        }
        if elapsed >= 4.0:
            self.data["armed"] = True
            self.data["gate"] = gate_snapshot(self.next_frame_id, timestamp_s=self.next_start_ms / 1000)


class FakeController:
    def __init__(self):
        self.commands = []
        self.arm_calls = 0

    def send_flight_command(self, command, command_allowed):
        if not command_allowed:
            return False
        self.commands.append(command)
        return True

    def arm(self):
        self.arm_calls += 1


def gate_snapshot(frame_id, timestamp_s=1.0):
    return {
        "frame_id": frame_id,
        "timestamp_s": timestamp_s,
        "detected": True,
        "frame_size": (640, 360),
        "centroid": (320.0, 180.0),
        "area_px": 10_000.0,
        "track_id": 1,
        "tracking_confidence": 1.0,
        "tracking_missed_frames": 0,
    }


class AIGPEnvTests(unittest.TestCase):
    def make_env(self, target_gate_count=None):
        data = {
            "race_status": {
                "sim_boot_time_ms": 1000,
                "race_start_boot_time_ms": 4000,
                "race_finish_time_ns": -1,
                "active_gate_index": 0,
                "received_at_s": -1.0,
            },
            "gate": gate_snapshot(1),
            "armed": True,
            "imu": {},
        }
        clock = FakeClock(data)
        controller = FakeController()
        manager = EpisodeManager(
            data,
            send_reset=clock.request_reset,
            send_arm=controller.arm,
            config=EpisodeConfig(target_gate_count=target_gate_count),
        )
        env = AIGPEnv(
            data,
            controller,
            manager,
            clock=clock,
            sleeper=clock.sleep,
        )
        return env, data, clock, controller

    def test_reset_waits_four_seconds_and_sends_no_flight_commands(self):
        env, _, clock, controller = self.make_env()

        observation, _ = env.reset()

        self.assertGreaterEqual(clock.now, 4.0)
        self.assertEqual(controller.commands, [])
        self.assertEqual(observation.shape, (38,))

    def test_step_holds_one_action_for_five_commands(self):
        env, _, clock, controller = self.make_env()
        env.reset()
        started = clock.now

        _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))

        self.assertAlmostEqual(clock.now - started, 0.1, places=6)
        self.assertEqual(len(controller.commands), 5)
        self.assertEqual(info["command_sends"], 5)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_curriculum_completion_returns_terminal_transition(self):
        env, data, _, controller = self.make_env(target_gate_count=1)
        env.reset()
        data["race_status"]["active_gate_index"] = 1

        _, reward, terminated, truncated, info = env.step(np.zeros(4))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "curriculum_complete")
        self.assertEqual(info["reward_components"]["gate_pass"], 20.0)
        self.assertGreater(reward, 19.0)
        self.assertEqual(controller.commands, [])

    def test_collision_terminates_and_requires_reset(self):
        env, data, _, _ = self.make_env()
        env.reset()
        data["collision"] = {
            "sequence": 1,
            "collision_id": 1002,
            "threat_level": 2,
        }

        _, _, terminated, truncated, info = env.step(np.zeros(4))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "environment_collision")
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(4))

    def test_sb3_environment_contract(self):
        env, _, _, _ = self.make_env()
        check_env(env, warn=True)

    def test_close_resets_an_active_simulator(self):
        env, _, clock, _ = self.make_env()
        env.reset()

        env.close()

        self.assertEqual(clock.reset_at, clock.now)


if __name__ == "__main__":
    unittest.main()
