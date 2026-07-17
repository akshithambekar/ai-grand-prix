import unittest

from controls.episode_manager import EpisodeConfig, EpisodeManager, EpisodePhase


def race(sim_ms=1000, start_ms=4000, gate=0, received_at_s=0.1, finish=-1):
    return {
        "sim_boot_time_ms": sim_ms,
        "race_start_boot_time_ms": start_ms,
        "race_finish_time_ns": finish,
        "active_gate_index": gate,
        "last_gate_race_time": -1,
        "received_at_s": received_at_s,
    }


def frame(frame_id, timestamp_s, detected=True):
    return {"frame_id": frame_id, "timestamp_s": timestamp_s, "detected": detected}


class EpisodeManagerTests(unittest.TestCase):
    def setUp(self):
        self.data = {"race_status": race(), "gate": frame(1, 1.0), "armed": True}
        self.resets = 0
        self.arms = 0

        def reset():
            self.resets += 1

        def arm():
            self.arms += 1

        self.manager = EpisodeManager(self.data, reset, arm)

    def start_episode(self):
        self.manager.request_reset(now=0.0)
        self.assertEqual(self.manager.update(now=0.1).phase, EpisodePhase.COUNTDOWN)
        self.data["race_status"] = race(sim_ms=4000, start_ms=4000, received_at_s=4.0)
        self.assertEqual(self.manager.update(now=4.0).phase, EpisodePhase.WAITING_FOR_ARM)
        self.assertEqual(self.manager.update(now=4.01).phase, EpisodePhase.AWAIT_FRESH_FRAME)
        self.data["gate"] = frame(2, 4.01)
        event = self.manager.update(now=4.1)
        self.assertTrue(event.episode_started)
        self.assertEqual(event.phase, EpisodePhase.ACTIVE)
        return event

    def test_countdown_disallows_commands_and_requires_fresh_frame(self):
        self.manager.request_reset(now=0.0)
        event = self.manager.update(now=0.1)
        self.assertEqual(event.phase, EpisodePhase.COUNTDOWN)
        self.assertFalse(event.command_allowed)

        self.data["race_status"] = race(sim_ms=4000, start_ms=4000, received_at_s=4.0)
        event = self.manager.update(now=4.0)
        self.assertEqual(event.phase, EpisodePhase.WAITING_FOR_ARM)
        self.assertFalse(event.command_allowed)

        event = self.manager.update(now=4.01)
        self.assertEqual(event.phase, EpisodePhase.AWAIT_FRESH_FRAME)
        self.data["gate"] = frame(2, 4.01)
        event = self.manager.update(now=4.1)
        self.assertTrue(event.command_allowed)
        self.assertTrue(event.episode_started)

    def test_four_second_post_reset_delay_is_always_enforced(self):
        self.data["armed"] = False
        self.manager.request_reset(now=0.0)
        self.manager.update(now=0.1)

        # Even if the simulator reports its countdown finished early, arming and
        # flight-control permission remain blocked until four seconds after reset.
        self.data["race_status"] = race(
            sim_ms=3000,
            start_ms=3000,
            received_at_s=3.0,
        )
        event = self.manager.update(now=3.0)
        self.assertEqual(event.phase, EpisodePhase.WAITING_FOR_ARM)
        event = self.manager.update(now=3.99)
        self.assertFalse(event.command_allowed)
        self.assertEqual(self.arms, 0)

        self.manager.update(now=4.0)
        self.assertEqual(self.arms, 1)

    def test_environment_collision_ends_episode_and_requests_reset(self):
        self.start_episode()
        self.data["collision"] = {
            "sequence": 1,
            "collision_id": 1002,
            "threat_level": 1,
            "impulse": 1.0,
        }
        event = self.manager.update(now=5.0)
        self.assertTrue(event.episode_ended)
        self.assertEqual(event.reason, "environment_collision")
        self.assertFalse(event.command_allowed)
        self.assertEqual(self.resets, 2)  # initial reset plus terminal reset

    def test_gate_timeout_ends_episode(self):
        self.start_episode()
        event = self.manager.update(now=4.1 + 15.0)
        self.assertTrue(event.episode_ended)
        self.assertEqual(event.reason, "gate_timeout")

    def test_gate_advance_resets_gate_timer_and_counts_pass(self):
        self.start_episode()
        self.data["race_status"] = race(sim_ms=5000, start_ms=4000, gate=1, received_at_s=5.0)
        event = self.manager.update(now=5.0)
        self.assertFalse(event.episode_ended)
        self.assertEqual(event.gates_passed, 1)
        self.assertEqual(event.phase, EpisodePhase.ACTIVE)


if __name__ == "__main__":
    unittest.main()
