import io
import time
import unittest

from rich.console import Console

from controls.training_dashboard import TrainingDashboardCallback


class TrainingDashboardTests(unittest.TestCase):
    def test_compact_dashboard_folds_core_metrics(self):
        console = Console(
            width=42,
            height=10,
            record=True,
            force_terminal=False,
            file=io.StringIO(),
        )
        dashboard = TrainingDashboardCallback(512, 5000, console=console)
        dashboard.started_at = time.monotonic()
        dashboard.step_reward = 1.25
        dashboard.active_episode_reward = 4.5
        dashboard.step_components = {"gate_pass": 20.0, "step": -0.01}
        dashboard.episode_components.update({"gate_pass": 20.0, "step": -0.4})
        dashboard.info = {
            "episode_id": 3,
            "active_gate_index": 1,
            "track_id": 7,
            "detected": True,
        }

        console.print(dashboard.render(width=42, height=10))
        output = console.export_text()

        self.assertIn("Losses", output)
        self.assertIn("Reward", output)
        self.assertIn("gate_pass", output)
        self.assertIn("episode=3", output)

    def test_wide_dashboard_has_organized_sections(self):
        console = Console(
            width=120,
            height=30,
            record=True,
            force_terminal=False,
            file=io.StringIO(),
        )
        dashboard = TrainingDashboardCallback(512, 5000, console=console)
        dashboard.started_at = time.monotonic()

        console.print(dashboard.render(width=120, height=30))
        output = console.export_text()

        self.assertIn("Optimization", output)
        self.assertIn("Reward breakdown", output)
        self.assertIn("Run and simulator", output)


if __name__ == "__main__":
    unittest.main()
