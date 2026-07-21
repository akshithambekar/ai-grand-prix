"""Validate live gate tracking during a bounded first-gate approach.

The probe performs settle, climb, and slow forward phases. It stops and resets after
the active gate advances, an episode termination condition fires, K is pressed, or
the horizontal target error exceeds its safety bound.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from pymavlink import mavutil


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controls.episode_manager import EpisodeConfig, EpisodeManager
from controls.mavlink_rx import MAVLinkRX
from controls.vision_rx import VisionRX
from scripts.reset_hotkey import ResetHotkey


CONTROL_HZ = 50.0
MAVLINK_CMD_SIM_RESET = 31000
RATES_ATTITUDE_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
LOG_DIR = REPO_ROOT / "artifacts" / "target_tracker_probe"


class TargetTrackerProbe:
    def __init__(self, connection, data, args):
        self.connection = connection
        self.data = data
        self.args = args
        self.system_boot_ms = int(time.time() * 1000)
        self.hotkey = ResetHotkey()
        self.started_at = None
        self.starting_gate = None
        self.stop_requested = False
        self.phase = "RESETTING"
        self.termination_reason = ""
        self.last_command = (0.0, 0.0, 0.0, 0.0)

        config = EpisodeConfig(
            gate_timeout_s=args.approach_timeout,
            episode_timeout_s=args.approach_timeout + 10.0,
        )
        self.episodes = EpisodeManager(
            data,
            send_reset=self.send_reset,
            send_arm=self.send_arm,
            config=config,
        )

        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = log_path.open("w", newline="", buffering=1)
        self.writer = csv.DictWriter(self.log_file, fieldnames=[
            "wall_time", "episode_phase", "probe_phase", "phase_elapsed",
            "active_gate_index", "frame_id", "detected", "track_id",
            "candidate_count", "tracking_confidence", "tracking_missed_frames",
            "track_switched", "track_switch_reason", "association_score",
            "rejected_edge_fragments", "selected_cx", "selected_cy",
            "selected_area_px", "candidates_json", "cmd_roll_rate",
            "cmd_pitch_rate", "cmd_yaw_rate", "cmd_thrust", "termination_reason",
        ])
        self.writer.writeheader()

    def run(self):
        print(f"Logging tracker probe to {self.args.log}", flush=True)
        self.episodes.request_reset(reason="tracker_probe_initial")
        next_tick = time.monotonic()
        try:
            while not self.stop_requested:
                now = time.monotonic()
                self._handle_hotkey()
                event = self.episodes.update(now=now)

                if event.episode_started:
                    self.started_at = now
                    status = self.data.get("race_status") or {}
                    self.starting_gate = status.get("active_gate_index", 0)
                    print("Tracker probe episode started.", flush=True)

                if event.episode_ended:
                    self.termination_reason = event.reason or "episode_ended"
                    print(f"Probe ended: {event.reason}", flush=True)
                    self.stop_requested = True

                if event.command_allowed and not self.stop_requested:
                    self._update_active(now)

                self._log_row(now, event.phase.value)
                next_tick += 1.0 / CONTROL_HZ
                time.sleep(max(0.0, next_tick - time.monotonic()))
        finally:
            self.hotkey.close()
            self.log_file.close()
            self._stop_receivers()

    def _update_active(self, now):
        status = self.data.get("race_status") or {}
        active_gate = status.get("active_gate_index")
        if (
            active_gate is not None
            and self.starting_gate is not None
            and active_gate > self.starting_gate
        ):
            print(f"Gate advanced {self.starting_gate} -> {active_gate}; probe passed.", flush=True)
            self.termination_reason = "tracker_probe_complete"
            self.episodes.request_reset(reason="tracker_probe_complete", now=now)
            self.stop_requested = True
            return

        gate = self.data.get("gate") or {}
        if self._horizontal_limit_exceeded(gate):
            print("Target exceeded horizontal safety bound; resetting.", flush=True)
            self.termination_reason = "horizontal_safety_bound"
            self.episodes.request_reset(reason="horizontal_safety_bound", now=now)
            self.stop_requested = True
            return

        elapsed = now - self.started_at
        if elapsed < self.args.settle_duration:
            self.phase = "SETTLE"
            command = (0.0, 0.0, 0.0, self.args.hover_thrust)
        elif elapsed < self.args.settle_duration + self.args.climb_duration:
            self.phase = "CLIMB"
            command = (
                0.0,
                0.0,
                0.0,
                self.args.hover_thrust + self.args.climb_thrust_delta,
            )
        else:
            self.phase = "APPROACH"
            approach_elapsed = elapsed - self.args.settle_duration - self.args.climb_duration
            pitch_rate = (
                self.args.forward_pitch_rate
                if approach_elapsed < self.args.pitch_pulse_duration
                else 0.0
            )
            if not gate.get("detected", False):
                pitch_rate = 0.0
            command = (
                0.0,
                pitch_rate,
                0.0,
                self.args.hover_thrust + self.args.approach_thrust_delta,
            )

        self.send_attitude_rates(*command)
        self.last_command = command

    def _horizontal_limit_exceeded(self, gate):
        if not gate.get("detected", False):
            return False
        centroid = gate.get("centroid")
        frame_size = gate.get("frame_size")
        if not centroid or not frame_size:
            return False
        normalized_error = 2.0 * (centroid[0] / frame_size[0]) - 1.0
        return abs(normalized_error) > self.args.horizontal_limit

    def send_reset(self):
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            MAVLINK_CMD_SIM_RESET,
            0,
            0, 0, 0, 0, 0, 0, 0,
        )

    def send_arm(self):
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0,
        )

    def send_attitude_rates(self, roll_rate, pitch_rate, yaw_rate, thrust):
        self.connection.mav.set_attitude_target_send(
            int(time.time() * 1000) - self.system_boot_ms,
            self.connection.target_system,
            self.connection.target_component,
            RATES_ATTITUDE_MASK,
            [1, 0, 0, 0],
            roll_rate,
            pitch_rate,
            yaw_rate,
            thrust,
        )

    def _handle_hotkey(self):
        if self.hotkey.consume_request():
            print("K pressed; resetting and stopping probe.", flush=True)
            self.termination_reason = "manual_probe_stop"
            self.episodes.request_reset(reason="manual_probe_stop")
            self.stop_requested = True

    def _log_row(self, now, episode_phase):
        gate = self.data.get("gate") or {}
        status = self.data.get("race_status") or {}
        centroid = gate.get("centroid") or (None, None)
        roll, pitch, yaw, thrust = self.last_command
        self.writer.writerow({
            "wall_time": time.time(),
            "episode_phase": episode_phase,
            "probe_phase": self.phase,
            "phase_elapsed": now - self.started_at if self.started_at is not None else "",
            "active_gate_index": status.get("active_gate_index", ""),
            "frame_id": gate.get("frame_id", ""),
            "detected": gate.get("detected", False),
            "track_id": gate.get("track_id", ""),
            "candidate_count": gate.get("candidate_count", 0),
            "tracking_confidence": gate.get("tracking_confidence", ""),
            "tracking_missed_frames": gate.get("tracking_missed_frames", ""),
            "track_switched": gate.get("track_switched", False),
            "track_switch_reason": gate.get("track_switch_reason", ""),
            "association_score": gate.get("association_score", ""),
            "rejected_edge_fragments": gate.get("rejected_edge_fragments", 0),
            "selected_cx": centroid[0] if centroid[0] is not None else "",
            "selected_cy": centroid[1] if centroid[1] is not None else "",
            "selected_area_px": gate.get("area_px", ""),
            "candidates_json": json.dumps(gate.get("candidates", ())),
            "cmd_roll_rate": roll,
            "cmd_pitch_rate": pitch,
            "cmd_yaw_rate": yaw,
            "cmd_thrust": thrust,
            "termination_reason": self.termination_reason,
        })

    def _stop_receivers(self):
        for key in ("mavlink_rx", "vision_rx"):
            receiver = self.data.get(key)
            if receiver is not None:
                receiver.get_thread_for_join().join(timeout=1.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="enable movement")
    parser.add_argument("--sim-ip", default="127.0.0.1")
    parser.add_argument("--sim-port", type=int, default=14550)
    parser.add_argument("--hover-thrust", type=float, default=0.265)
    parser.add_argument("--climb-thrust-delta", type=float, default=0.015)
    parser.add_argument("--approach-thrust-delta", type=float, default=0.005)
    parser.add_argument("--settle-duration", type=float, default=0.5)
    parser.add_argument("--climb-duration", type=float, default=1.0)
    parser.add_argument("--forward-pitch-rate", type=float, default=-0.05)
    parser.add_argument("--pitch-pulse-duration", type=float, default=0.5)
    parser.add_argument("--approach-timeout", type=float, default=12.0)
    parser.add_argument("--horizontal-limit", type=float, default=0.30)
    parser.add_argument(
        "--log",
        default=str(LOG_DIR / f"target_tracker_probe_{int(time.time())}.csv"),
    )
    return parser.parse_args()


def validate_args(args):
    if not args.execute:
        raise SystemExit("Dry run only. Add --execute to send movement commands.")
    if args.settle_duration < 0 or args.climb_duration < 0:
        raise SystemExit("settle and climb durations must be non-negative")
    if args.pitch_pulse_duration < 0 or args.approach_timeout <= 0:
        raise SystemExit("pulse duration must be non-negative and timeout must be positive")
    if not 0.0 <= args.hover_thrust + args.climb_thrust_delta <= 1.0:
        raise SystemExit("climb thrust must stay within [0, 1]")
    if not 0.0 <= args.hover_thrust + args.approach_thrust_delta <= 1.0:
        raise SystemExit("approach thrust must stay within [0, 1]")
    if abs(args.forward_pitch_rate) > 0.25:
        raise SystemExit("forward pitch rate magnitude must not exceed 0.25 rad/s")
    if not 0.0 < args.horizontal_limit <= 1.0:
        raise SystemExit("horizontal limit must be in (0, 1]")


def main():
    args = parse_args()
    validate_args(args)
    connection = mavutil.mavlink_connection(f"udpin:{args.sim_ip}:{args.sim_port}")
    print("Waiting for simulator heartbeat...", flush=True)
    connection.wait_heartbeat()
    data = {"mavlink_rx": None, "vision_rx": None}
    data["mavlink_rx"] = MAVLinkRX.create_mavlink_rx(connection, data)
    data["vision_rx"] = VisionRX(data)
    try:
        TargetTrackerProbe(connection, data, args).run()
    finally:
        connection.close()


if __name__ == "__main__":
    main()
