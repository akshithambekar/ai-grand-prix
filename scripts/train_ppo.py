"""Train direct-action PPO against one official Windows simulator instance."""

import argparse
import csv
import math
import os
import sys
from collections import Counter, deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from dotenv import load_dotenv
from tqdm.auto import tqdm

from controls.episode_manager import EpisodeConfig
from controls.model_compat import stamp_model_schema, validate_model_schema
from controls.runtime import create_official_env


ARTIFACTS = REPO_ROOT / "artifacts" / "state_v2"
PPO_DEVICE = "cpu"
EPISODE_FIELDS = [
    "episode_id", "episode_duration_s", "episode_steps", "gates_passed",
    "highest_gate_index", "reset_reason", "collision_count", "detection_rate",
    "mean_center_error", "control_rate_hz", "vision_rate_hz",
    "vehicle_state_availability", "track_geometry_availability", "state_fallback_count",
    "mean_speed_m_s", "max_speed_m_s", "max_attitude_rad", "max_body_rate_rad_s",
    "gate_distance_change_m", "position_min_ned", "position_max_ned",
    "mean_vision_range_error_m",
]


class EpisodeCSVCallback(BaseCallback):
    def __init__(self, path):
        super().__init__()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=EPISODE_FIELDS)
        self._writer.writeheader()

    def _on_step(self):
        for done, info in zip(self.locals.get("dones", ()), self.locals.get("infos", ())):
            if done:
                self._writer.writerow({field: info.get(field) for field in EPISODE_FIELDS})
        return True

    def _on_training_end(self):
        self._file.close()


class TrainingProgressCallback(BaseCallback):
    """Live run-local PPO progress, including rollouts, epochs, and episodes."""

    def __init__(self, requested_timesteps, checkpoint_freq):
        super().__init__()
        self.requested_timesteps = requested_timesteps
        self.checkpoint_freq = checkpoint_freq
        self.episodes = 0
        self.reset_reasons = Counter()
        self.last_reset_reason = "-"
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.gates_passed = 0
        self._initial_updates = 0
        self._last_timesteps = 0
        self._progress = None

    def _on_training_start(self):
        rollout_size = self.model.n_steps * self.training_env.num_envs
        expected_updates = math.ceil(self.requested_timesteps / rollout_size)
        effective_timesteps = expected_updates * rollout_size
        self._initial_updates = self.model._n_updates
        self._last_timesteps = self.num_timesteps
        self._progress = tqdm(
            total=effective_timesteps,
            desc="PPO training",
            unit="step",
            dynamic_ncols=True,
            mininterval=0.5,
        )
        minibatches_per_epoch = math.ceil(rollout_size / self.model.batch_size)
        tqdm.write(
            f"PPO schedule: {rollout_size:,} steps/update, {self.model.n_epochs} "
            f"epochs/update, {minibatches_per_epoch} minibatches/epoch, "
            f"{expected_updates} policy updates."
        )
        if effective_timesteps != self.requested_timesteps:
            tqdm.write(
                f"Requested {self.requested_timesteps:,} timesteps; PPO collects complete "
                f"{rollout_size:,}-step rollouts, so this run will collect {effective_timesteps:,}."
            )
        self._refresh_stats()

    def _on_rollout_start(self):
        # The previous PPO optimization has completed by the next rollout start.
        self._refresh_stats()

    def _on_step(self):
        delta = self.num_timesteps - self._last_timesteps
        if delta > 0:
            self._progress.update(delta)
            self._last_timesteps = self.num_timesteps
        for done, info in zip(self.locals.get("dones", ()), self.locals.get("infos", ())):
            if done:
                self.episodes += 1
                reason = info.get("reset_reason") or "unknown"
                self.last_reset_reason = str(reason)
                self.reset_reasons[self.last_reset_reason] += 1
                self.gates_passed += int(info.get("gates_passed") or 0)
                episode = info.get("episode", {})
                if "r" in episode:
                    self.episode_rewards.append(float(episode["r"]))
                if "l" in episode:
                    self.episode_lengths.append(int(episode["l"]))
        self._refresh_stats()
        return True

    def _refresh_stats(self):
        if self._progress is None:
            return
        epochs = max(0, self.model._n_updates - self._initial_updates)
        n_epochs = self.model.n_epochs
        policy_updates = epochs // n_epochs
        rollout_size = self.model.n_steps * self.training_env.num_envs
        expected_updates = math.ceil(self.requested_timesteps / rollout_size)
        minibatches_per_epoch = math.ceil(rollout_size / self.model.batch_size)
        checkpoint_period = self.checkpoint_freq * self.training_env.num_envs
        checkpoint_in = checkpoint_period - (self.num_timesteps % checkpoint_period)
        mean_reward = (
            sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0
        )
        mean_episode_length = (
            sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0.0
        )
        approximate_kl = self.model.logger.name_to_value.get("train/approx_kl", 0.0)
        self._progress.set_postfix(
            updates=f"{policy_updates}/{expected_updates}",
            epochs=f"{epochs}/{expected_updates * n_epochs}",
            minibatches=(
                f"{epochs * minibatches_per_epoch}/"
                f"{expected_updates * n_epochs * minibatches_per_epoch}"
            ),
            episodes=self.episodes,
            reward100=f"{mean_reward:.2f}",
            ep_len100=f"{mean_episode_length:.1f}",
            gates=self.gates_passed,
            kl=f"{float(approximate_kl):.4f}",
            last_reset=self.last_reset_reason,
            checkpoint_in=checkpoint_in,
            refresh=False,
        )

    def _on_training_end(self):
        # The final optimization occurs after the final rollout callback.
        self._refresh_stats()
        if self._progress is not None:
            self._progress.close()


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _env_int(name, default):
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


def parse_args(argv=None):
    load_dotenv(REPO_ROOT / ".env")
    try:
        execute = _env_bool("AIGP_EXECUTE", False)
        sim_port = _env_int("AIGP_SIM_PORT", 14550)
        vision_port = _env_int("AIGP_VISION_PORT", 5600)
        target_gates = _env_int("AIGP_TARGET_GATES", 1)
        total_timesteps = _env_int("AIGP_TOTAL_TIMESTEPS", 100_000)
        seed = _env_int("AIGP_SEED", 0)
        n_steps = _env_int("AIGP_N_STEPS", 512)
        batch_size = _env_int("AIGP_BATCH_SIZE", 64)
        n_epochs = _env_int("AIGP_N_EPOCHS", 10)
        checkpoint_freq = _env_int("AIGP_CHECKPOINT_FREQ", 5_000)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    resume = os.getenv("AIGP_RESUME", "").strip()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action=argparse.BooleanOptionalAction, default=execute,
        help="enable live training (env: AIGP_EXECUTE)",
    )
    parser.add_argument("--sim-ip", default=os.getenv("AIGP_SIM_IP", "127.0.0.1"))
    parser.add_argument("--sim-port", type=int, default=sim_port)
    parser.add_argument("--vision-port", type=int, default=vision_port)
    parser.add_argument("--target-gates", type=int, default=target_gates, help="0 means full course")
    parser.add_argument("--total-timesteps", type=int, default=total_timesteps)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--n-steps", type=int, default=n_steps)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--n-epochs", type=int, default=n_epochs)
    parser.add_argument("--checkpoint-freq", type=int, default=checkpoint_freq)
    parser.add_argument("--resume", type=Path, default=Path(resume) if resume else None)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if not args.execute:
        raise SystemExit("Add --execute to connect and train on the official simulator.")
    if args.target_gates < 0 or min(
        args.total_timesteps, args.n_steps, args.batch_size, args.n_epochs, args.checkpoint_freq
    ) <= 0:
        raise SystemExit("target gates must be non-negative and all training counts must be positive")
    if args.batch_size > args.n_steps:
        raise SystemExit("batch size cannot exceed n_steps for this single-environment trainer")
    print(f"PPO device: {PPO_DEVICE}", flush=True)

    for directory in ("models", "checkpoints", "tensorboard"):
        (ARTIFACTS / directory).mkdir(parents=True, exist_ok=True)

    env = create_official_env(
        sim_ip=args.sim_ip,
        sim_port=args.sim_port,
        vision_port=args.vision_port,
        episode_config=EpisodeConfig(
            target_gate_count=args.target_gates or None,
        ),
    )
    monitored = Monitor(env, filename=str(ARTIFACTS / "monitor.csv"))
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(ARTIFACTS / "checkpoints"),
            name_prefix="ppo_aigp",
        ),
        EpisodeCSVCallback(ARTIFACTS / "episodes.csv"),
        TrainingProgressCallback(args.total_timesteps, args.checkpoint_freq),
    ])

    try:
        if args.resume:
            model = validate_model_schema(PPO.load(args.resume, device=PPO_DEVICE))
            model.set_env(monitored)
        else:
            model = PPO(
                "MlpPolicy",
                monitored,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                learning_rate=3e-4,
                gamma=0.995,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.005,
                policy_kwargs={"net_arch": [128, 128]},
                tensorboard_log=str(ARTIFACTS / "tensorboard"),
                seed=args.seed,
                verbose=1,
                device=PPO_DEVICE,
            )
        model.verbose = 0
        stamp_model_schema(model)
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
        output = ARTIFACTS / "models" / "ppo_aigp_final"
        model.save(output)
        print(f"Model saved to {output}.zip", flush=True)
    finally:
        monitored.close()


if __name__ == "__main__":
    main()
