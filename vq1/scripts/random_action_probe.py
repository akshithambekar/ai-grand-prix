"""Run a small number of bounded random Gym episodes for live safety validation."""

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controls.runtime import create_official_env


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim-ip", default="127.0.0.1")
    parser.add_argument("--sim-port", type=int, default=14550)
    parser.add_argument("--vision-port", type=int, default=5600)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.execute:
        raise SystemExit("Add --execute to send bounded random actions.")
    if args.episodes <= 0:
        raise SystemExit("episodes must be positive")

    env = create_official_env(
        sim_ip=args.sim_ip,
        sim_port=args.sim_port,
        vision_port=args.vision_port,
    )
    env.action_space.seed(args.seed)
    reasons = Counter()
    try:
        for episode in range(1, args.episodes + 1):
            env.reset()
            while True:
                _, _, terminated, truncated, info = env.step(env.action_space.sample())
                if terminated or truncated:
                    break
            reason = info.get("termination_reason")
            reasons[reason] += 1
            print(f"Episode {episode}: {reason}", flush=True)
    finally:
        env.close()
    print(f"Reset reasons: {dict(reasons)}", flush=True)


if __name__ == "__main__":
    main()
