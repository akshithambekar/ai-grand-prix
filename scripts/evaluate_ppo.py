"""Run deterministic evaluation using the single official simulator instance."""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO

from controls.episode_manager import EpisodeConfig
from controls.runtime import create_official_env


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sim-ip", default="127.0.0.1")
    parser.add_argument("--sim-port", type=int, default=14550)
    parser.add_argument("--vision-port", type=int, default=5600)
    parser.add_argument("--target-gates", type=int, default=1, help="0 means full course")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.execute:
        raise SystemExit("Add --execute to run live evaluation.")
    if args.episodes <= 0 or args.target_gates < 0:
        raise SystemExit("episodes must be positive and target gates non-negative")

    env = create_official_env(
        sim_ip=args.sim_ip,
        sim_port=args.sim_port,
        vision_port=args.vision_port,
        episode_config=EpisodeConfig(target_gate_count=args.target_gates or None),
    )
    model = PPO.load(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reasons = Counter()
    successes = 0
    fields = [
        "episode", "reward", "gates_passed", "highest_gate_index",
        "duration_s", "reason", "success",
    ]

    try:
        with args.output.open("w", newline="", buffering=1) as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for episode in range(1, args.episodes + 1):
                observation, _ = env.reset()
                total_reward = 0.0
                while True:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        break
                success = info.get("termination_reason") in {
                    "curriculum_complete",
                    "course_complete",
                }
                successes += int(success)
                reasons[info.get("termination_reason")] += 1
                writer.writerow({
                    "episode": episode,
                    "reward": total_reward,
                    "gates_passed": info.get("gates_passed"),
                    "highest_gate_index": info.get("highest_gate_index"),
                    "duration_s": info.get("episode_duration_s"),
                    "reason": info.get("termination_reason"),
                    "success": success,
                })
                print(f"Episode {episode}: {info.get('termination_reason')}", flush=True)
    finally:
        env.close()

    print(f"Success rate: {successes / args.episodes:.0%}", flush=True)
    print(f"Reset reasons: {dict(reasons)}", flush=True)


if __name__ == "__main__":
    main()
