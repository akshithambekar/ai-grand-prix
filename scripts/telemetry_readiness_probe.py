"""Validate restored state_v2 telemetry before training or evaluation."""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controls.readiness import telemetry_report, wait_for_vehicle_state
from controls.setup import setup_components, shutdown_components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-ip", default="127.0.0.1")
    parser.add_argument("--sim-port", type=int, default=14550)
    parser.add_argument("--vision-port", type=int, default=5600)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    data = {}
    components = setup_components(
        data, int(time.time() * 1000), args.sim_ip, args.sim_port,
        vision_port=args.vision_port,
    )
    try:
        wait_for_vehicle_state(data, timeout_s=args.timeout)
        time.sleep(0.5)
        report = telemetry_report(data)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        if not report["track_geometry_valid"]:
            print("WARNING: track geometry is unavailable; policy will use vision fallback.", flush=True)
    finally:
        shutdown_components(components)


if __name__ == "__main__":
    main()
