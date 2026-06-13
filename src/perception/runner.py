"""Live perception runner entrypoint."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from pymavlink import mavutil

from perception.main import PerceptionRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live perception against the simulator.")
    parser.add_argument("--sim-host", default="127.0.0.1")
    parser.add_argument("--mavlink-port", type=int, default=14550)
    parser.add_argument("--vision-host", default="0.0.0.0")
    parser.add_argument("--vision-port", type=int, default=5600)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("artifacts/perception/calibration/camera_calibration.json"),
    )
    parser.add_argument("--weights", type=Path, default=Path("runs/perception/latest/best.pt"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    conn = mavutil.mavlink_connection(f"udpin:{args.sim_host}:{args.mavlink_port}")
    print("Waiting for heartbeat...")
    conn.wait_heartbeat()
    print(f"Connected to system {conn.target_system}")

    weights = args.weights if args.weights.exists() else None

    runner = PerceptionRunner(
        mavlink_connection=conn,
        vision_host=args.vision_host,
        vision_port=args.vision_port,
        calibration_path=args.calibration,
        detector_weights_path=weights,
        detector_device=args.device,
    )

    runner.start()
    print("Perception runner started. Press Ctrl+C to stop.")

    try:
        while True:
            result = runner.get_latest_result()
            if result is not None:
                print(
                    {
                        "frame_id": result.frame_id,
                        "telemetry_timestamp_ns": result.telemetry_timestamp_ns,
                        "active_gate_index": result.active_gate_index,
                        "inference_path": result.inference_path,
                        "detection_confidence": result.detection_confidence,
                        "projection_confidence": result.projection_confidence,
                        "bbox_xyxy": result.bbox_xyxy,
                        "failure_reason": result.failure_reason,
                    }
                )
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
        print("Perception runner stopped.")
