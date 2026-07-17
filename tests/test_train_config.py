import os
import unittest
from unittest.mock import patch

from scripts.train_ppo import PPO_DEVICE, PPO_POLICY_KWARGS, parse_args, training_episode_config


class TrainingConfigurationTests(unittest.TestCase):
    def test_ppo_device_is_cpu(self):
        self.assertEqual(PPO_DEVICE, "cpu")

    def test_initial_exploration_standard_deviation_is_reduced(self):
        self.assertEqual(PPO_POLICY_KWARGS["log_std_init"], -1.0)

    def test_environment_defaults_are_loaded(self):
        values = {
            "AIGP_EXECUTE": "true",
            "AIGP_SIM_IP": "10.0.0.2",
            "AIGP_SIM_PORT": "14551",
            "AIGP_VISION_PORT": "5601",
            "AIGP_TARGET_GATES": "3",
            "AIGP_TOTAL_TIMESTEPS": "250000",
            "AIGP_SEED": "7",
            "AIGP_RESUME": "model.zip",
            "AIGP_N_STEPS": "1024",
            "AIGP_BATCH_SIZE": "128",
            "AIGP_N_EPOCHS": "5",
            "AIGP_CHECKPOINT_FREQ": "10000",
            "AIGP_DASHBOARD": "false",
            "AIGP_DASHBOARD_REFRESH_HZ": "2.5",
        }
        with patch.dict(os.environ, values, clear=False):
            args = parse_args([])
        self.assertTrue(args.execute)
        self.assertEqual(args.sim_ip, "10.0.0.2")
        self.assertEqual(args.sim_port, 14551)
        self.assertEqual(args.vision_port, 5601)
        self.assertEqual(args.target_gates, 3)
        self.assertEqual(args.total_timesteps, 250000)
        self.assertEqual(args.seed, 7)
        self.assertEqual(str(args.resume), "model.zip")
        self.assertEqual(args.n_steps, 1024)
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(args.n_epochs, 5)
        self.assertEqual(args.checkpoint_freq, 10000)
        self.assertFalse(args.dashboard)
        self.assertEqual(args.dashboard_refresh_hz, 2.5)

    def test_cli_overrides_environment_and_can_disable_execution(self):
        with patch.dict(os.environ, {"AIGP_EXECUTE": "true", "AIGP_TARGET_GATES": "3"}):
            args = parse_args(["--no-execute", "--target-gates", "0", "--no-dashboard"])
        self.assertFalse(args.execute)
        self.assertEqual(args.target_gates, 0)
        self.assertFalse(args.dashboard)

    def test_first_gate_curriculum_uses_permissive_exploration_limits(self):
        config = training_episode_config(1)
        self.assertEqual(config.target_gate_count, 1)
        self.assertEqual(config.gate_timeout_s, 25.0)
        self.assertEqual(config.no_detection_timeout_s, 5.0)
        self.assertEqual(config.stuck_grace_s + config.stuck_hold_s, 15.0)
        self.assertEqual(config.divergence_distance_m, 8.0)
        self.assertEqual(config.divergence_hold_s, 2.0)

    def test_full_course_keeps_standard_safety_limits(self):
        config = training_episode_config(0)
        self.assertIsNone(config.target_gate_count)
        self.assertEqual(config.gate_timeout_s, 15.0)
        self.assertEqual(config.no_detection_timeout_s, 2.0)


if __name__ == "__main__":
    unittest.main()
