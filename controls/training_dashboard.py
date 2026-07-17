import math
import time
from collections import Counter, defaultdict, deque

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from stable_baselines3.common.callbacks import BaseCallback


LOSS_KEYS = (
    ("loss", "train/loss"),
    ("value loss", "train/value_loss"),
    ("policy loss", "train/policy_gradient_loss"),
    ("entropy", "train/entropy_loss"),
    ("approx KL", "train/approx_kl"),
    ("clip fraction", "train/clip_fraction"),
)


def _format_number(value, precision=4):
    if value is None:
        return "waiting"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{precision}f}" if math.isfinite(number) else str(number)


def _folded_pairs(values):
    if not values:
        return "waiting"
    return "  ".join(f"{key}={_format_number(value, 3)}" for key, value in values.items())


class TrainingDashboardCallback(BaseCallback):
    """Responsive live view of PPO optimization, rewards, and simulator state."""

    def __init__(
        self,
        requested_timesteps,
        checkpoint_freq,
        *,
        refresh_hz=4.0,
        enabled=True,
        console=None,
    ):
        super().__init__()
        self.requested_timesteps = requested_timesteps
        self.checkpoint_freq = checkpoint_freq
        self.refresh_hz = refresh_hz
        self.enabled = enabled
        self.console = console or Console()
        self.live = None
        self.started_at = None
        self.last_refresh_at = 0.0
        self.effective_timesteps = requested_timesteps
        self.expected_updates = 0
        self.initial_updates = 0

        self.step_reward = 0.0
        self.active_episode_reward = 0.0
        self.last_episode_reward = None
        self.mean_rewards = deque(maxlen=100)
        self.mean_lengths = deque(maxlen=100)
        self.step_components = {}
        self.episode_components = defaultdict(float)
        self.last_episode_components = {}
        self.info = {}
        self.episodes = 0
        self.gates_passed = 0
        self.reset_reasons = Counter()

    def _on_training_start(self):
        rollout_size = self.model.n_steps * self.training_env.num_envs
        self.expected_updates = math.ceil(self.requested_timesteps / rollout_size)
        self.effective_timesteps = self.expected_updates * rollout_size
        self.initial_updates = self.model._n_updates
        self.started_at = time.monotonic()
        if self.enabled:
            self.live = Live(
                self.render(),
                console=self.console,
                refresh_per_second=self.refresh_hz,
                vertical_overflow="visible",
                transient=False,
            )
            self.live.start()

    def _on_rollout_start(self):
        self._refresh(force=True)

    def _on_step(self):
        rewards = self.locals.get("rewards", ())
        infos = self.locals.get("infos", ())
        dones = self.locals.get("dones", ())
        if len(rewards):
            self.step_reward = float(rewards[0])
            self.active_episode_reward += self.step_reward
        if len(infos):
            self.info = infos[0]
            self.step_components = dict(self.info.get("reward_components", {}))
            for name, value in self.step_components.items():
                self.episode_components[name] += float(value)
        if len(dones) and dones[0]:
            self.episodes += 1
            self.gates_passed += int(self.info.get("gates_passed") or 0)
            reason = str(self.info.get("reset_reason") or "unknown")
            self.reset_reasons[reason] += 1
            episode = self.info.get("episode", {})
            final_reward = float(episode.get("r", self.active_episode_reward))
            self.last_episode_reward = final_reward
            self.mean_rewards.append(final_reward)
            if "l" in episode:
                self.mean_lengths.append(int(episode["l"]))
            self.last_episode_components = dict(self.episode_components)
            self.active_episode_reward = 0.0
            self.episode_components.clear()

        self._refresh()
        return True

    def _on_training_end(self):
        self._refresh(force=True)
        self.close()

    def close(self):
        if self.live is not None:
            self.live.stop()
            self.live = None

    def _refresh(self, force=False):
        if self.live is None:
            return
        now = time.monotonic()
        if force or now - self.last_refresh_at >= 1.0 / self.refresh_hz:
            self.live.update(self.render(), refresh=True)
            self.last_refresh_at = now

    def render(self, width=None, height=None):
        width = width or self.console.size.width
        height = height or self.console.size.height
        return self._render_compact() if width < 78 or height < 27 else self._render_wide()

    def _loss_values(self):
        logger = getattr(getattr(self, "model", None), "logger", None)
        values = getattr(logger, "name_to_value", {})
        return {label: values.get(key) for label, key in LOSS_KEYS}

    def _training_values(self):
        current = int(getattr(self, "num_timesteps", 0) or 0)
        elapsed = max(time.monotonic() - self.started_at, 1e-9) if self.started_at else 0.0
        model = getattr(self, "model", None)
        training_env = model.get_env() if model is not None else None
        rollout_size = max(
            int(getattr(model, "n_steps", 1))
            * int(getattr(training_env, "num_envs", 1)),
            1,
        )
        epochs = max(0, int(getattr(model, "_n_updates", 0)) - self.initial_updates)
        n_epochs = max(int(getattr(model, "n_epochs", 1)), 1)
        checkpoint_period = max(self.checkpoint_freq, 1)
        return {
            "steps": f"{current:,}/{self.effective_timesteps:,}",
            "progress": f"{100 * current / max(self.effective_timesteps, 1):.1f}%",
            "rate": f"{current / elapsed:.1f}/s" if elapsed else "waiting",
            "rollout": f"{current % rollout_size}/{rollout_size}",
            "updates": f"{epochs // n_epochs}/{self.expected_updates}",
            "epochs": str(epochs),
            "checkpoint in": str(checkpoint_period - (current % checkpoint_period)),
        }

    def _render_compact(self):
        train = self._training_values()
        losses = _folded_pairs(self._loss_values())
        step_breakdown = _folded_pairs(self.step_components)
        episode_breakdown = _folded_pairs(self.episode_components)
        last_episode_breakdown = _folded_pairs(self.last_episode_components)
        last_reward = _format_number(self.last_episode_reward, 3)
        lines = [
            Text(
                f"{train['steps']}  {train['progress']}  {train['rate']}  "
                f"updates {train['updates']}",
                style="bold cyan",
                overflow="fold",
            ),
            Text(f"Losses  {losses}", overflow="fold"),
            Text(
                f"Reward  step={self.step_reward:.3f}  "
                f"active={self.active_episode_reward:.3f}  last={last_reward}",
                style="bold green",
                overflow="fold",
            ),
            Text(f"Step breakdown  {step_breakdown}", overflow="fold"),
            Text(f"Episode totals  {episode_breakdown}", overflow="fold"),
            Text(f"Last episode totals  {last_episode_breakdown}", overflow="fold"),
            Text(
                "State  "
                f"episode={self.info.get('episode_id', '-')}  "
                f"gate={self.info.get('active_gate_index', '-')}  "
                f"track={self.info.get('track_id', '-')}  "
                f"detected={self.info.get('detected', '-')}",
                overflow="fold",
            ),
        ]
        return Panel(
            Group(*lines),
            title="AI Grand Prix PPO",
            border_style="bright_blue",
            padding=(0, 1),
        )

    def _render_wide(self):
        container = Table.grid(expand=True)
        container.add_column(ratio=1)
        container.add_column(ratio=1)

        optimization = Table(box=box.SIMPLE, expand=True, title="Optimization")
        optimization.add_column("Metric", style="cyan", no_wrap=True)
        optimization.add_column("Value", justify="right")
        for label, value in self._loss_values().items():
            optimization.add_row(label, _format_number(value))
        optimization.add_row("step reward", _format_number(self.step_reward, 3))
        optimization.add_row("active reward", _format_number(self.active_episode_reward, 3))
        optimization.add_row("last reward", _format_number(self.last_episode_reward, 3))

        run = Table(box=box.SIMPLE, expand=True, title="Run and simulator")
        run.add_column("Metric", style="magenta", no_wrap=True)
        run.add_column("Value", justify="right")
        for label, value in self._training_values().items():
            run.add_row(label, value)
        run.add_row("episodes", str(self.episodes))
        run.add_row("gates", str(self.gates_passed))
        run.add_row("active gate", str(self.info.get("active_gate_index", "-")))
        run.add_row("track", str(self.info.get("track_id", "-")))
        run.add_row("detected", str(self.info.get("detected", "-")))
        container.add_row(optimization, run)

        breakdown = Table(box=box.SIMPLE, expand=True, title="Reward breakdown")
        breakdown.add_column("Component", style="green")
        breakdown.add_column("Current step", justify="right")
        breakdown.add_column("Active episode", justify="right")
        breakdown.add_column("Last episode", justify="right")
        names = sorted(
            set(self.step_components)
            | set(self.episode_components)
            | set(self.last_episode_components)
        )
        if names:
            for name in names:
                breakdown.add_row(
                    name,
                    _format_number(self.step_components.get(name, 0.0), 3),
                    _format_number(self.episode_components.get(name, 0.0), 3),
                    _format_number(self.last_episode_components.get(name, 0.0), 3),
                )
        else:
            breakdown.add_row("waiting for first transition", "-", "-", "-")

        mean_reward = sum(self.mean_rewards) / len(self.mean_rewards) if self.mean_rewards else None
        footer = Text(
            f"episodes={self.episodes}  reward100={_format_number(mean_reward, 2)}  "
            f"resets={dict(self.reset_reasons) or '{}'}",
            style="yellow",
            overflow="fold",
        )
        return Panel(
            Group(container, breakdown, footer),
            title="AI Grand Prix PPO Training",
            border_style="bright_blue",
            padding=(0, 1),
        )
