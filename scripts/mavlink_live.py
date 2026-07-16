"""Display live AI Grand Prix MAVLink telemetry in a terminal dashboard.

Run this while the simulator is sending MAVLink to the selected UDP port. Only one
process should bind that port, so stop ``main.py`` first or configure a separate
telemetry destination/port.
"""

import argparse
import struct
import sys
import time

from pymavlink import mavutil


RACE_STATUS_DATA_TYPE = 1
RACE_STATUS_FORMAT = "<BQqqIq"
DEFAULT_SIM_IP = "127.0.0.1"
DEFAULT_SIM_PORT = 14550


class TelemetryState:
    def __init__(self):
        self.started_at = time.monotonic()
        self.message_counts = {}
        self.last_seen = {}
        self.heartbeat = None
        self.imu = None
        self.race_status = None
        self.collision = None
        self.actuators = None

    def update(self, msg):
        now = time.monotonic()
        msg_type = msg.get_type()
        self.message_counts[msg_type] = self.message_counts.get(msg_type, 0) + 1
        self.last_seen[msg_type] = now

        if msg_type == "HEARTBEAT":
            self.heartbeat = {
                "armed": bool(
                    msg.base_mode
                    & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                ),
                "base_mode": msg.base_mode,
                "custom_mode": msg.custom_mode,
                "system_status": msg.system_status,
            }
        elif msg_type == "HIGHRES_IMU":
            self.imu = {
                "time_usec": msg.time_usec,
                "accel": (msg.xacc, msg.yacc, msg.zacc),
                "gyro": (msg.xgyro, msg.ygyro, msg.zgyro),
            }
        elif msg_type == "ENCAPSULATED_DATA":
            self._update_encapsulated(msg)
        elif msg_type == "COLLISION":
            self.collision = {
                "id": msg.id,
                "threat": msg.threat_level,
                "impulse": msg.horizontal_minimum_delta,
                "received_at": now,
            }
        elif msg_type == "ACTUATOR_OUTPUT_STATUS":
            self.actuators = tuple(msg.actuator[:4])

    def _update_encapsulated(self, msg):
        payload = bytes(msg.data)
        if not payload or payload[0] != RACE_STATUS_DATA_TYPE:
            return

        required_size = struct.calcsize(RACE_STATUS_FORMAT)
        if len(payload) < required_size:
            return

        (
            _,
            sim_boot_time_ms,
            race_start_boot_time_ms,
            race_finish_time_ns,
            active_gate_index,
            last_gate_race_time,
        ) = struct.unpack_from(RACE_STATUS_FORMAT, payload)
        self.race_status = {
            "sim_boot_time_ms": sim_boot_time_ms,
            "race_start_boot_time_ms": race_start_boot_time_ms,
            "race_finish_time_ns": race_finish_time_ns,
            "active_gate_index": active_gate_index,
            "last_gate_race_time": last_gate_race_time,
        }

    def render(self, connection):
        now = time.monotonic()
        lines = [
            "AI Grand Prix MAVLink Live",
            "=" * 56,
            (
                f"System: {connection.target_system}  "
                f"Component: {connection.target_component}  "
                f"Monitor uptime: {now - self.started_at:7.1f}s"
            ),
        ]

        if self.heartbeat is None:
            lines.append("Heartbeat: waiting")
        else:
            heartbeat_age = now - self.last_seen["HEARTBEAT"]
            lines.append(
                f"Heartbeat: {'ARMED' if self.heartbeat['armed'] else 'disarmed'}  "
                f"base={self.heartbeat['base_mode']}  "
                f"custom={self.heartbeat['custom_mode']}  "
                f"status={self.heartbeat['system_status']}  "
                f"age={heartbeat_age:.2f}s"
            )

        lines.extend(self._render_race())
        lines.extend(self._render_imu())

        if self.actuators is None:
            lines.append("Actuators: waiting")
        else:
            values = "  ".join(f"{value:+.3f}" for value in self.actuators)
            lines.append(f"Actuators [FL FR BL BR]: {values}")

        if self.collision is None:
            lines.append("Collision: none observed")
        else:
            collision_age = now - self.collision["received_at"]
            lines.append(
                f"Collision: id={self.collision['id']}  "
                f"threat={self.collision['threat']}  "
                f"impulse={self.collision['impulse']:.3f} kg m/s  "
                f"age={collision_age:.1f}s"
            )

        lines.append("")
        lines.append("Messages received:")
        if not self.message_counts:
            lines.append("  waiting")
        else:
            for msg_type in sorted(self.message_counts):
                age = now - self.last_seen[msg_type]
                lines.append(
                    f"  {msg_type:<28} "
                    f"count={self.message_counts[msg_type]:>8}  age={age:6.2f}s"
                )

        return "\n".join(lines)

    def _render_race(self):
        race = self.race_status
        if race is None:
            return ["Race: waiting for race status"]

        sim_ms = race["sim_boot_time_ms"]
        start_ms = race["race_start_boot_time_ms"]
        finish_ns = race["race_finish_time_ns"]

        if finish_ns >= 0:
            phase = "FINISHED"
        elif start_ms < 0:
            phase = "WAITING"
        elif sim_ms < start_ms:
            phase = f"COUNTDOWN {(start_ms - sim_ms) / 1000.0:.2f}s"
        else:
            phase = "RACING"

        return [
            (
                f"Race: {phase}  gate={race['active_gate_index']}  "
                f"sim={sim_ms / 1000.0:.3f}s  start_ms={start_ms}  "
                f"last_gate={race['last_gate_race_time']}"
            )
        ]

    def _render_imu(self):
        if self.imu is None:
            return ["IMU: waiting"]

        ax, ay, az = self.imu["accel"]
        gx, gy, gz = self.imu["gyro"]
        return [
            f"Accel m/s^2: x={ax:+8.3f}  y={ay:+8.3f}  z={az:+8.3f}",
            f"Gyro rad/s:  x={gx:+8.3f}  y={gy:+8.3f}  z={gz:+8.3f}",
        ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-ip", default=DEFAULT_SIM_IP)
    parser.add_argument("--sim-port", type=int, default=DEFAULT_SIM_PORT)
    parser.add_argument("--refresh-hz", type=float, default=5.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.refresh_hz <= 0:
        raise SystemExit("--refresh-hz must be greater than zero")

    connection = mavutil.mavlink_connection(
        f"udpin:{args.sim_ip}:{args.sim_port}"
    )
    state = TelemetryState()
    refresh_period = 1.0 / args.refresh_hz
    next_render = 0.0

    print("Waiting for MAVLink messages. Press Ctrl+C to exit.", flush=True)
    try:
        while True:
            msg = connection.recv_match(blocking=False)
            if msg is not None and msg.get_type() != "BAD_DATA":
                state.update(msg)

            now = time.monotonic()
            if now >= next_render:
                dashboard = state.render(connection)
                if sys.stdout.isatty():
                    print("\033[2J\033[H", end="")
                print(dashboard, flush=True)
                next_render = now + refresh_period

            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nStopping MAVLink monitor.", flush=True)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
