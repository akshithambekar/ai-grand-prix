"""Run bounded control pulses and log the resulting MAVLink and vision data.

The probe is deliberately opt-in because it moves the simulated drone. Start it
with ``--execute`` after stopping any other process that binds UDP ports 14550 and
5600. Press K during a run to issue a reset and stop the probe.
"""

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from pymavlink import mavutil


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controls.mavlink_rx import MAVLinkRX
from controls.vision_rx import VisionRX
from scripts.reset_hotkey import ResetHotkey


CONTROL_HZ = 50.0
MAVLINK_CMD_SIM_RESET = 31000
RATES_ATTITUDE_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE


@dataclass(frozen=True)
class ProbePhase:
    name: str
    duration_s: float
    roll_rate: float
    pitch_rate: float
    yaw_rate: float
    thrust: float


class ControlProbe:
    def __init__(self, connection, data, args):
        self.connection = connection
        self.data = data
        self.args = args
        self.system_boot_ms = int(time.time() * 1000)
        self.hotkey = ResetHotkey()
        self.reset_sent_at = None
        self.reset_seen_status = False
        self.reset_reference_start = None
        self.phase_index = 0
        self.phase_started_at = None
        self.phase_collision_sequence = 0
        self.state = "RESETTING"
        self.stop_requested = False
        self.last_arm_sent_at = 0.0
        self.log_file = open(args.log, "w", newline="", buffering=1)
        self.writer = csv.DictWriter(self.log_file, fieldnames=[
            "wall_time", "state", "phase", "phase_elapsed",
            "sim_boot_time_ms", "race_start_boot_time_ms", "active_gate_index",
            "armed", "cmd_roll_rate", "cmd_pitch_rate", "cmd_yaw_rate", "cmd_thrust",
            "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
            "gate_detected", "gate_cx", "gate_cy", "gate_area_px", "gate_range_m",
            "collision_id", "collision_threat", "collision_impulse",
        ])
        self.writer.writeheader()

    def phases(self):
        h = self.args.hover_thrust
        step = self.args.thrust_step
        rate = self.args.pulse_rate
        duration = self.args.phase_duration
        return [
            ProbePhase("thrust_low", duration, 0.0, 0.0, 0.0, h - step),
            ProbePhase("thrust_hover", duration, 0.0, 0.0, 0.0, h),
            ProbePhase("thrust_high", duration, 0.0, 0.0, 0.0, h + step),
            ProbePhase("roll_positive", duration, rate, 0.0, 0.0, h),
            ProbePhase("roll_negative", duration, -rate, 0.0, 0.0, h),
            ProbePhase("pitch_positive", duration, 0.0, rate, 0.0, h),
            ProbePhase("pitch_negative", duration, 0.0, -rate, 0.0, h),
            ProbePhase("yaw_positive", duration, 0.0, 0.0, rate, h),
            ProbePhase("yaw_negative", duration, 0.0, 0.0, -rate, h),
        ]

    def run(self):
        print(f"Logging probe data to {self.args.log}", flush=True)
        print("Resetting into a fresh simulator countdown...", flush=True)
        self.request_reset()
        next_tick = time.monotonic()
        try:
            while not self.stop_requested:
                now = time.monotonic()
                self.handle_hotkey()
                self.update_state(now)
                self.log_row(now)

                if self.state == "RUNNING":
                    self.send_phase_command()
                elif self.state in {"RESETTING", "COUNTDOWN", "WAITING_FOR_ARM"}:
                    # No attitude/rate/thrust command is sent during reset or countdown.
                    pass

                next_tick += 1.0 / CONTROL_HZ
                time.sleep(max(0.0, next_tick - time.monotonic()))
        finally:
            self.hotkey.close()
            self.log_file.close()
            self.stop_receiver_threads()

    def request_reset(self):
        status = self.data.get("race_status") or {}
        self.reset_reference_start = status.get("race_start_boot_time_ms")
        self.reset_sent_at = time.monotonic()
        self.reset_seen_status = False
        self.state = "RESETTING"
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            MAVLINK_CMD_SIM_RESET,
            0,
            0, 0, 0, 0, 0, 0, 0,
        )

    def update_state(self, now):
        status = self.data.get("race_status")
        if self.state == "RESETTING":
            if status and status.get("received_at_s", 0.0) >= self.reset_sent_at:
                self.reset_seen_status = True
                start_ms = status.get("race_start_boot_time_ms", -1)
                sim_ms = status.get("sim_boot_time_ms", -1)
                if start_ms >= 0 and sim_ms < start_ms:
                    self.state = "COUNTDOWN"
                elif start_ms >= 0 and sim_ms >= start_ms:
                    self.state = "WAITING_FOR_ARM"
            return

        if self.state == "COUNTDOWN":
            if status and status.get("race_start_boot_time_ms", -1) >= 0:
                if status["sim_boot_time_ms"] >= status["race_start_boot_time_ms"]:
                    self.state = "WAITING_FOR_ARM"
            return

        if self.state == "WAITING_FOR_ARM":
            if self.data.get("armed"):
                self.phase_started_at = now
                # Reset/launch collisions may remain in shared_data. Only events
                # received after this phase becomes active can terminate it.
                collision = self.data.get("collision") or {}
                self.phase_collision_sequence = collision.get("sequence", 0)
                self.state = "RUNNING"
                print(f"Starting phase: {self.phases()[self.phase_index].name}", flush=True)
            elif now - self.last_arm_sent_at >= 1.0:
                self.send_arm()
                self.last_arm_sent_at = now
            return

        if self.state == "RUNNING":
            if self.data.get("collision"):
                collision = self.data["collision"]
                is_new_collision = (
                    collision.get("sequence", 0) > self.phase_collision_sequence
                )
                if is_new_collision and collision.get("received_at_s", 0.0) >= self.phase_started_at:
                    print("Collision observed; stopping probe.", flush=True)
                    self.request_reset()
                    self.stop_requested = True
                    return

            if now - self.phase_started_at >= self.phases()[self.phase_index].duration_s:
                self.phase_index += 1
                if self.phase_index >= len(self.phases()):
                    print("Probe complete; resetting simulator.", flush=True)
                    self.request_reset()
                    self.stop_requested = True
                else:
                    print(f"Resetting before phase {self.phases()[self.phase_index].name}", flush=True)
                    self.request_reset()

    def handle_hotkey(self):
        if self.hotkey.consume_request():
            print("K pressed; resetting and stopping probe.", flush=True)
            self.request_reset()
            self.stop_requested = True

    def send_arm(self):
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0,
        )

    def send_phase_command(self):
        phase = self.phases()[self.phase_index]
        self.connection.mav.set_attitude_target_send(
            int(time.time() * 1000) - self.system_boot_ms,
            self.connection.target_system,
            self.connection.target_component,
            RATES_ATTITUDE_MASK,
            [1, 0, 0, 0],
            phase.roll_rate,
            phase.pitch_rate,
            phase.yaw_rate,
            phase.thrust,
        )

    def log_row(self, now):
        status = self.data.get("race_status") or {}
        imu = self.data.get("imu") or {}
        gate = self.data.get("gate") or {}
        centroid = gate.get("centroid") or (None, None)
        collision = self.data.get("collision") or {}
        phase = self.phases()[self.phase_index] if self.phase_index < len(self.phases()) else None
        self.writer.writerow({
            "wall_time": time.time(),
            "state": self.state,
            "phase": phase.name if phase else "complete",
            "phase_elapsed": (now - self.phase_started_at) if self.phase_started_at else "",
            "sim_boot_time_ms": status.get("sim_boot_time_ms", ""),
            "race_start_boot_time_ms": status.get("race_start_boot_time_ms", ""),
            "active_gate_index": status.get("active_gate_index", ""),
            "armed": self.data.get("armed", False),
            "cmd_roll_rate": phase.roll_rate if phase else 0.0,
            "cmd_pitch_rate": phase.pitch_rate if phase else 0.0,
            "cmd_yaw_rate": phase.yaw_rate if phase else 0.0,
            "cmd_thrust": phase.thrust if phase else 0.0,
            "accel_x": imu.get("xacc", ""), "accel_y": imu.get("yacc", ""),
            "accel_z": imu.get("zacc", ""), "gyro_x": imu.get("xgyro", ""),
            "gyro_y": imu.get("ygyro", ""), "gyro_z": imu.get("zgyro", ""),
            "gate_detected": gate.get("detected", False),
            "gate_cx": centroid[0] if centroid[0] is not None else "",
            "gate_cy": centroid[1] if centroid[1] is not None else "",
            "gate_area_px": gate.get("area_px", ""),
            "gate_range_m": gate.get("range_m", ""),
            "collision_id": collision.get("collision_id", ""),
            "collision_threat": collision.get("threat_level", ""),
            "collision_impulse": collision.get("impulse", ""),
        })

    def stop_receiver_threads(self):
        for receiver in (self.data.get("mavlink_rx"), self.data.get("vision_rx")):
            if receiver is not None:
                receiver.get_thread_for_join().join(timeout=1.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="enable movement")
    parser.add_argument("--sim-ip", default="127.0.0.1")
    parser.add_argument("--sim-port", type=int, default=14550)
    parser.add_argument("--hover-thrust", type=float, default=0.265)
    parser.add_argument("--thrust-step", type=float, default=0.015)
    parser.add_argument("--pulse-rate", type=float, default=0.05)
    parser.add_argument("--phase-duration", type=float, default=0.75)
    parser.add_argument("--log", default=f"control_probe_{int(time.time())}.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.execute:
        print("Dry run only. Add --execute to send movement commands.")
        return
    if args.phase_duration <= 0:
        raise SystemExit("--phase-duration must be greater than zero")
    if args.pulse_rate < 0 or args.pulse_rate > 0.25:
        raise SystemExit("--pulse-rate must be between 0 and 0.25 rad/s")
    if args.hover_thrust - args.thrust_step < 0 or args.hover_thrust + args.thrust_step > 1:
        raise SystemExit("hover thrust +/- thrust step must stay within [0, 1]")

    connection = mavutil.mavlink_connection(f"udpin:{args.sim_ip}:{args.sim_port}")
    print("Waiting for simulator heartbeat...", flush=True)
    connection.wait_heartbeat()
    data = {"mavlink_rx": None, "vision_rx": None}
    data["mavlink_rx"] = MAVLinkRX.create_mavlink_rx(connection, data)
    data["vision_rx"] = VisionRX(data)
    try:
        ControlProbe(connection, data, args).run()
    finally:
        connection.close()


if __name__ == "__main__":
    main()
