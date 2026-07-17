"""Train direct-action PPO against one official Windows simulator instance."""

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from dotenv import load_dotenv

from controls.episode_manager import EpisodeConfig
from controls.model_compat import stamp_model_schema, validate_model_schema
from controls.runtime import create_official_env
from controls.training_dashboard import TrainingDashboardCallback


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
        self.close()

    def close(self):
        if not self._file.closed:
            self._file.close()


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


def _env_float(name, default):
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, got {value!r}") from error


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
        dashboard = _env_bool("AIGP_DASHBOARD", True)
        dashboard_refresh_hz = _env_float("AIGP_DASHBOARD_REFRESH_HZ", 4.0)
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
    parser.add_argument(
        "--dashboard", action=argparse.BooleanOptionalAction, default=dashboard,
        help="show the responsive live dashboard (env: AIGP_DASHBOARD)",
    )
    parser.add_argument(
        "--dashboard-refresh-hz", type=float, default=dashboard_refresh_hz,
        help="dashboard refresh rate (env: AIGP_DASHBOARD_REFRESH_HZ)",
    )
    parser.add_argument("--resume", type=Path, default=Path(resume) if resume else None)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if not args.execute:
        raise SystemExit("Add --execute to connect and train on the official simulator.")
    if args.target_gates < 0 or args.dashboard_refresh_hz <= 0 or min(
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
    episode_csv = EpisodeCSVCallback(ARTIFACTS / "episodes.csv")
    dashboard = TrainingDashboardCallback(
        args.total_timesteps,
        args.checkpoint_freq,
        refresh_hz=args.dashboard_refresh_hz,
        enabled=args.dashboard,
    )
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(ARTIFACTS / "checkpoints"),
            name_prefix="ppo_aigp",
        ),
        episode_csv,
        dashboard,
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
                verbose=0,
                device=PPO_DEVICE,
            )
        model.verbose = 0
        stamp_model_schema(model)
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
        output = ARTIFACTS / "models" / "ppo_aigp_final"
        model.save(output)
        print(f"Model saved to {output}.zip", flush=True)
    finally:
        dashboard.close()
        episode_csv.close()
        monitored.close()


if __name__ == "__main__":
    main()
