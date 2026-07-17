"""Run deterministic evaluation using the single official simulator instance."""

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO

from controls.episode_manager import EpisodeConfig
from controls.model_compat import validate_model_schema
from controls.observation import OBSERVATION_SCHEMA
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
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts") / OBSERVATION_SCHEMA / "evaluation.csv",
    )
    parser.add_argument(
        "--trace-output", type=Path,
        default=Path("artifacts") / OBSERVATION_SCHEMA / "evaluation_trace.csv",
    )
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
    model = validate_model_schema(PPO.load(args.model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    reasons = Counter()
    successes = 0
    fields = [
        "episode", "reward", "gates_passed", "highest_gate_index",
        "duration_s", "reason", "success", "vehicle_state_availability",
        "track_geometry_availability", "state_fallback_count", "mean_speed_m_s",
        "max_speed_m_s", "max_attitude_rad", "max_body_rate_rad_s",
        "gate_distance_change_m", "mean_vision_range_error_m",
    ]
    trace_fields = [
        "episode", "step", "time_s", "reward", "termination_reason",
        "action_roll", "action_pitch", "action_yaw", "action_thrust",
        "detected", "gate_phase", "plane_distance_m", "signed_plane_distance_m",
        "lateral_m", "vertical_m", "alignment_error",
        "position_n", "position_e", "position_d",
        "velocity_n", "velocity_e", "velocity_d", "roll", "pitch", "yaw",
        "step_cost", "aligned_approach", "alignment_progress", "descent_stability",
        "altitude_progress", "attitude_stability", "rate_slew", "thrust_slew",
        "gate_pass", "collision", "terminal_failure",
    ]

    try:
        with (
            args.output.open("w", newline="", buffering=1) as output,
            args.trace_output.open("w", newline="", buffering=1) as trace_output,
        ):
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            trace_writer = csv.DictWriter(trace_output, fieldnames=trace_fields)
            trace_writer.writeheader()
            for episode in range(1, args.episodes + 1):
                observation, _ = env.reset()
                total_reward = 0.0
                step = 0
                while True:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    step += 1
                    total_reward += reward
                    trace_writer.writerow(_trace_row(episode, step, action, reward, info))
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
                    "vehicle_state_availability": info.get("vehicle_state_availability"),
                    "track_geometry_availability": info.get("track_geometry_availability"),
                    "state_fallback_count": info.get("state_fallback_count"),
                    "mean_speed_m_s": info.get("mean_speed_m_s"),
                    "max_speed_m_s": info.get("max_speed_m_s"),
                    "max_attitude_rad": info.get("max_attitude_rad"),
                    "max_body_rate_rad_s": info.get("max_body_rate_rad_s"),
                    "gate_distance_change_m": info.get("gate_distance_change_m"),
                    "mean_vision_range_error_m": info.get("mean_vision_range_error_m"),
                })
                print(f"Episode {episode}: {info.get('termination_reason')}", flush=True)
    finally:
        env.close()

    print(f"Success rate: {successes / args.episodes:.0%}", flush=True)
    print(f"Reset reasons: {dict(reasons)}", flush=True)


def _trace_row(episode, step, action, reward, info):
    vehicle = info.get("vehicle_state") or {}
    gate = info.get("active_gate_state") or {}
    components = info.get("reward_components") or {}
    position = vehicle.get("position_ned") or (None, None, None)
    velocity = vehicle.get("velocity_ned") or (None, None, None)
    euler = vehicle.get("euler") or (None, None, None)
    lateral = gate.get("lateral_m")
    vertical = gate.get("vertical_m")
    width = gate.get("width_m")
    height = gate.get("height_m")
    alignment_error = None
    try:
        alignment_error = math.hypot(
            float(lateral) / max(float(width) / 2.0, 0.1),
            float(vertical) / max(float(height) / 2.0, 0.1),
        )
    except (TypeError, ValueError):
        pass
    plane_distance = gate.get("plane_distance_m")
    gate_phase = "commit" if (
        gate.get("valid") and plane_distance is not None
        and float(plane_distance) <= 3.0
        and alignment_error is not None and alignment_error <= 1.25
    ) else "approach"
    names = {
        "step_cost": "step",
        "aligned_approach": "aligned_approach",
        "alignment_progress": "alignment_progress",
        "descent_stability": "descent_stability",
        "altitude_progress": "altitude_progress",
        "attitude_stability": "attitude_stability",
        "rate_slew": "rate_slew",
        "thrust_slew": "thrust_slew",
        "gate_pass": "gate_pass",
        "collision": "collision",
        "terminal_failure": "terminal_failure",
    }
    return {
        "episode": episode,
        "step": step,
        "time_s": step / 10.0,
        "reward": reward,
        "termination_reason": info.get("termination_reason"),
        "action_roll": action[0], "action_pitch": action[1],
        "action_yaw": action[2], "action_thrust": action[3],
        "detected": info.get("detected"), "gate_phase": gate_phase,
        "plane_distance_m": plane_distance,
        "signed_plane_distance_m": gate.get("signed_plane_distance_m"),
        "lateral_m": lateral, "vertical_m": vertical,
        "alignment_error": alignment_error,
        "position_n": position[0], "position_e": position[1], "position_d": position[2],
        "velocity_n": velocity[0], "velocity_e": velocity[1], "velocity_d": velocity[2],
        "roll": euler[0], "pitch": euler[1], "yaw": euler[2],
        **{output: components.get(source, 0.0) for output, source in names.items()},
    }


if __name__ == "__main__":
    main()
