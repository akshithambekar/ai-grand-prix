"""Exercise the Gym pipeline with the first-gate command sequence proven in flight."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controls.controller import FlightCommand
from controls.episode_manager import EpisodeConfig, EpisodePhase
from controls.runtime import create_official_env


DEFAULT_LOG_DIR = REPO_ROOT / "artifacts" / "gym_control_probe"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="enable live movement")
    parser.add_argument("--sim-ip", default="127.0.0.1")
    parser.add_argument("--sim-port", type=int, default=14550)
    parser.add_argument("--vision-port", type=int, default=5600)
    parser.add_argument("--settle-duration", type=float, default=0.5)
    parser.add_argument("--climb-duration", type=float, default=2.5)
    parser.add_argument("--pitch-pulse-duration", type=float, default=0.5)
    parser.add_argument("--approach-timeout", type=float, default=12.0)
    parser.add_argument("--post-pass-duration", type=float, default=1.0)
    parser.add_argument(
        "--log",
        default=str(DEFAULT_LOG_DIR / f"gym_control_probe_{int(time.time())}.csv"),
    )
    return parser.parse_args()


def command_for_elapsed(elapsed, args, detected, passed):
    if passed or elapsed < args.settle_duration:
        return FlightCommand(0.0, 0.0, 0.0, 0.265)
    if elapsed < args.settle_duration + args.climb_duration:
        return FlightCommand(0.0, 0.0, 0.0, 0.280)
    approach_elapsed = elapsed - args.settle_duration - args.climb_duration
    pitch = -0.05 if detected and approach_elapsed < args.pitch_pulse_duration else 0.0
    return FlightCommand(0.0, pitch, 0.0, 0.270)


def main():
    args = parse_args()
    if not args.execute:
        raise SystemExit("Add --execute to connect and send flight commands.")

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = create_official_env(
        sim_ip=args.sim_ip,
        sim_port=args.sim_port,
        vision_port=args.vision_port,
        episode_config=EpisodeConfig(
            gate_timeout_s=args.approach_timeout,
            episode_timeout_s=args.approach_timeout + 15.0,
        ),
    )
    fields = [
        "wall_time", "elapsed", "phase", "active_gate_index", "track_id",
        "detected", "action", "reward", "reward_components", "terminated",
        "truncated", "termination_reason", "command_sends", "observation",
    ]

    try:
        observation, info = env.reset()
        starting_gate = info.get("active_gate_index", 0)
        started_at = time.monotonic()
        passed_at = None
        print(f"Episode active at gate {starting_gate}; running Gym probe.", flush=True)

        with log_path.open("w", newline="", buffering=1) as log_file:
            writer = csv.DictWriter(log_file, fieldnames=fields)
            writer.writeheader()
            while True:
                elapsed = time.monotonic() - started_at
                command = command_for_elapsed(
                    elapsed,
                    args,
                    bool(observation[0] > 0.5),
                    passed_at is not None,
                )
                action = env.action_mapper.action_for_command(command)
                observation, reward, terminated, truncated, info = env.step(action)
                gate_index = info.get("active_gate_index")
                if passed_at is None and gate_index is not None and gate_index > starting_gate:
                    passed_at = time.monotonic()
                    print(f"Gate advanced {starting_gate} -> {gate_index}.", flush=True)

                writer.writerow({
                    "wall_time": time.time(),
                    "elapsed": elapsed,
                    "phase": env.episodes.phase.value,
                    "active_gate_index": gate_index,
                    "track_id": info.get("track_id"),
                    "detected": info.get("detected"),
                    "action": json.dumps(action.tolist()),
                    "reward": reward,
                    "reward_components": json.dumps(info.get("reward_components", {})),
                    "terminated": terminated,
                    "truncated": truncated,
                    "termination_reason": info.get("termination_reason"),
                    "command_sends": info.get("command_sends"),
                    "observation": json.dumps(observation.tolist()),
                })

                if terminated or truncated:
                    print(f"Probe ended: {info.get('termination_reason')}", flush=True)
                    break
                if passed_at is not None and time.monotonic() - passed_at >= args.post_pass_duration:
                    print("Post-pass tracker observation complete.", flush=True)
                    break
                if elapsed >= args.approach_timeout:
                    print("Probe script timeout reached.", flush=True)
                    break

        if env.episodes.phase == EpisodePhase.ACTIVE:
            env.episodes.request_reset(reason="gym_probe_complete")
        print(f"Log written to {log_path}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
