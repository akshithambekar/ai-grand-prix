"""Train direct-action PPO against one official Windows simulator instance."""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from controls.episode_manager import EpisodeConfig
from controls.runtime import create_official_env


ARTIFACTS = REPO_ROOT / "artifacts"
EPISODE_FIELDS = [
    "episode_id", "episode_duration_s", "episode_steps", "gates_passed",
    "highest_gate_index", "reset_reason", "collision_count", "detection_rate",
    "mean_center_error", "control_rate_hz", "vision_rate_hz",
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="enable live training")
    parser.add_argument("--sim-ip", default="127.0.0.1")
    parser.add_argument("--sim-port", type=int, default=14550)
    parser.add_argument("--vision-port", type=int, default=5600)
    parser.add_argument("--target-gates", type=int, default=1, help="0 means full course")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.execute:
        raise SystemExit("Add --execute to connect and train on the official simulator.")
    if args.target_gates < 0 or args.total_timesteps <= 0:
        raise SystemExit("target gates must be non-negative and timesteps must be positive")

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
            save_freq=5_000,
            save_path=str(ARTIFACTS / "checkpoints"),
            name_prefix="ppo_aigp",
        ),
        EpisodeCSVCallback(ARTIFACTS / "episodes.csv"),
    ])

    try:
        if args.resume:
            model = PPO.load(args.resume, env=monitored)
        else:
            model = PPO(
                "MlpPolicy",
                monitored,
                n_steps=512,
                batch_size=64,
                n_epochs=10,
                learning_rate=3e-4,
                gamma=0.995,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.005,
                policy_kwargs={"net_arch": [128, 128]},
                tensorboard_log=str(ARTIFACTS / "tensorboard"),
                seed=args.seed,
                verbose=1,
            )
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
        output = ARTIFACTS / "models" / "ppo_aigp_final"
        model.save(output)
        print(f"Model saved to {output}.zip", flush=True)
    finally:
        monitored.close()


if __name__ == "__main__":
    main()
