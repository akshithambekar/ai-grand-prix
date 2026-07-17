"""Replay a target-tracker probe CSV through the PPO reward calculator."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controls.observation import ObservationEncoder
from controls.reward import RewardCalculator


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def optional_int(value):
    return int(value) if value not in (None, "") else None


def row_snapshot(row, width, height):
    detected = row.get("detected") == "True"
    gate_index = optional_int(row.get("active_gate_index"))
    track_id = optional_int(row.get("track_id"))
    gate = {
        "detected": detected,
        "frame_size": (width, height),
        "centroid": (
            float(row["selected_cx"]),
            float(row["selected_cy"]),
        ) if detected and row.get("selected_cx") and row.get("selected_cy") else None,
        "area_px": float(row["selected_area_px"]) if detected and row.get("selected_area_px") else None,
        "track_id": track_id,
        "tracking_confidence": float(row.get("tracking_confidence") or 0.0),
        "tracking_missed_frames": int(row.get("tracking_missed_frames") or 0),
    }
    return {
        "gate": gate,
        "race_status": {"active_gate_index": gate_index},
        "imu": {},
    }


def main():
    args = parse_args()
    output_path = args.output or args.input.with_name(f"{args.input.stem}_rewards.csv")
    encoder = ObservationEncoder()
    rewards = RewardCalculator()
    previous_action = np.zeros(4, dtype=np.float32)
    total = 0.0
    gate_bonuses = 0

    with args.input.open(newline="") as source, output_path.open("w", newline="") as output:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(
            output,
            fieldnames=["row", "active_gate_index", "track_id", "reward", "components"],
        )
        writer.writeheader()
        initialized = False
        for index, row in enumerate(reader):
            data = row_snapshot(row, args.width, args.height)
            observation = encoder.encode(data, previous_action)
            if not initialized:
                rewards.reset(data, observation)
                initialized = True
                continue
            result = rewards.compute(data, observation, row.get("termination_reason") or None)
            gate_bonuses += int("gate_pass" in result.components)
            total += result.total
            writer.writerow({
                "row": index,
                "active_gate_index": row.get("active_gate_index"),
                "track_id": row.get("track_id"),
                "reward": result.total,
                "components": dict(result.components),
            })

    print(f"Total reward: {total:.3f}", flush=True)
    print(f"Gate-bonus transitions: {gate_bonuses}", flush=True)
    print(f"Output written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
