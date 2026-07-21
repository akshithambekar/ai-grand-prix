"""Synchronized MAVLink, FPV, key, and command evidence capture."""

from __future__ import annotations

import ctypes
import json
import math
import os
import socket
import struct
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from pymavlink import mavutil


MAVLINK_CMD_SIM_RESET = 31000
RATES_ATTITUDE_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
VELOCITY_POSITION_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)
FRAME_HEADER = struct.Struct("<IHHIIQ")
RACE_STATUS = struct.Struct("<BQqqIq")
TRACK_CHUNK_HEADER = struct.Struct("<BH")
TRACK_GATE = struct.Struct("<Hfffffffff")
CYAN_LOWER_HSV = np.array((78, 70, 80), dtype=np.uint8)
CYAN_UPPER_HSV = np.array((105, 255, 255), dtype=np.uint8)
CYAN_PRESENT_FRACTION = 0.0002


def host_clock() -> dict[str, int]:
    return {
        "host_time_ns": time.time_ns(),
        "host_monotonic_ns": time.monotonic_ns(),
    }


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (bytes, bytearray)):
        return list(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


class EvidenceWriter:
    """Write all event types into one host-monotonic JSONL timeline."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir.resolve()
        self.frames_dir = self.run_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._stream = (self.run_dir / "events.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        self._lock = threading.Lock()
        self.counts = Counter()

    def write(self, event_type: str, **payload: Any) -> None:
        event = {"event": event_type, **host_clock(), **json_safe(payload)}
        encoded = json.dumps(event, separators=(",", ":"), allow_nan=False)
        with self._lock:
            self._stream.write(encoded + "\n")
            self.counts[event_type] += 1

    def close(self) -> None:
        with self._lock:
            self._stream.close()


class TelemetryRecorder:
    """Receive every MAVLink message and retain selected live snapshots."""

    def __init__(self, connection, writer: EvidenceWriter):
        self.connection = connection
        self.writer = writer
        self.message_counts = Counter()
        self._latest: dict[str, dict[str, Any]] = {}
        self._track_chunks: dict[int, dict[int, bytes]] = {}
        self._expected_track_chunks: dict[int, int] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._started = False
        self._thread = threading.Thread(
            target=self._receive_loop, name="mavlink-evidence", daemon=True
        )

    def start(self) -> None:
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)

    def latest(self, message_type: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._latest.get(message_type)
            return dict(value) if value is not None else None

    def _receive_loop(self) -> None:
        while not self._stop.is_set():
            try:
                message = self.connection.recv_match(blocking=False)
            except ConnectionResetError as exc:
                self.writer.write("receiver_error", source="mavlink", error=str(exc))
                return
            if message is None:
                time.sleep(0.001)
                continue
            message_type = message.get_type()
            if message_type == "BAD_DATA":
                continue
            fields = json_safe(message.to_dict())
            self.message_counts[message_type] += 1
            with self._lock:
                self._latest[message_type] = fields
            self.writer.write("mavlink", message_type=message_type, fields=fields)
            if message_type == "ENCAPSULATED_DATA":
                self._record_race_status(message)
                self._record_track_chunk(message)
            elif message_type == "DATA_TRANSMISSION_HANDSHAKE":
                transfer_id = int(message.width)
                self._track_chunks[transfer_id] = {}
                self._expected_track_chunks[transfer_id] = int(message.packets)
                self.writer.write(
                    "track_handshake",
                    transfer_id=transfer_id,
                    expected_chunks=int(message.packets),
                )

    def _record_race_status(self, message) -> None:
        raw = bytes(message.data)
        if len(raw) < RACE_STATUS.size or raw[0] != 1:
            return
        values = RACE_STATUS.unpack_from(raw)
        self.writer.write(
            "race_status",
            data_type=values[0],
            sim_boot_time_ms=values[1],
            race_start_boot_time_ms=values[2],
            race_finish_time_ns=values[3],
            active_gate_index=values[4],
            last_gate_race_time=values[5],
        )

    def _record_track_chunk(self, message) -> None:
        raw = bytes(message.data)
        if len(raw) < TRACK_CHUNK_HEADER.size or raw[0] != 2:
            return
        _, transfer_id = TRACK_CHUNK_HEADER.unpack_from(raw)
        if transfer_id not in self._expected_track_chunks:
            self.writer.write("track_error", reason="chunk_without_handshake", transfer_id=transfer_id)
            return
        self._track_chunks[transfer_id][int(message.seqnr)] = raw[TRACK_CHUNK_HEADER.size:]
        expected = self._expected_track_chunks[transfer_id]
        if len(self._track_chunks[transfer_id]) != expected:
            return
        try:
            payload = b"".join(
                self._track_chunks[transfer_id][index] for index in range(expected)
            )
        except KeyError:
            self.writer.write("track_error", reason="missing_chunk", transfer_id=transfer_id)
            return
        del self._track_chunks[transfer_id]
        del self._expected_track_chunks[transfer_id]
        self._record_track_data(transfer_id, payload)

    def _record_track_data(self, transfer_id: int, payload: bytes) -> None:
        if len(payload) < 2:
            self.writer.write("track_error", reason="short_payload", transfer_id=transfer_id)
            return
        gate_count = struct.unpack_from("<H", payload)[0]
        offset = 2
        gates = []
        for _ in range(gate_count):
            if len(payload) - offset < TRACK_GATE.size:
                self.writer.write(
                    "track_error",
                    reason="short_gate_payload",
                    transfer_id=transfer_id,
                    parsed_gates=len(gates),
                    expected_gates=gate_count,
                )
                return
            values = TRACK_GATE.unpack_from(payload, offset)
            offset += TRACK_GATE.size
            geometry = values[1:]
            geometry_published = any(
                math.isfinite(value) and abs(value) > 1e-9 for value in geometry
            )
            gates.append(
                {
                    "gate_id": values[0],
                    "position_ned": values[1:4],
                    "orientation_ned_wxyz": values[4:8],
                    "width": values[8],
                    "height": values[9],
                    "geometry_published": geometry_published,
                }
            )
        snapshot = {
            "transfer_id": transfer_id,
            "gate_count": gate_count,
            "geometry_published": any(gate["geometry_published"] for gate in gates),
            "gates": gates,
        }
        with self._lock:
            self._latest["TRACK_DATA"] = snapshot
        self.writer.write("track_data", **snapshot)


def cyan_fraction(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, CYAN_LOWER_HSV, CYAN_UPPER_HSV)
    return float(np.count_nonzero(mask)) / float(mask.size)


class FrameRecorder:
    """Reassemble every UDP JPEG and save it with simulator and host clocks."""

    def __init__(self, bind_ip: str, bind_port: int, writer: EvidenceWriter):
        self.bind_ip = bind_ip
        self.bind_port = bind_port
        self.writer = writer
        self.frame_count = 0
        self.decode_failures = 0
        self.frames_with_cyan = 0
        self.cyan_fraction_max = 0.0
        self._stop = threading.Event()
        self._started = False
        self._thread = threading.Thread(
            target=self._receive_loop, name="fpv-evidence", daemon=True
        )
        self._socket: socket.socket | None = None

    def start(self) -> None:
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        if self._socket is not None:
            self._socket.close()
        self._thread.join(timeout=2.0)

    def summary(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "decode_failures": self.decode_failures,
            "frames_with_cyan": self.frames_with_cyan,
            "cyan_fraction_max": self.cyan_fraction_max,
            "cyan_present": self.frames_with_cyan > 0,
        }

    def _receive_loop(self) -> None:
        frames: dict[int, dict[str, Any]] = {}
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket = sock
        sock.settimeout(0.25)
        try:
            sock.bind((self.bind_ip, self.bind_port))
        except OSError as exc:
            self.writer.write("receiver_error", source="fpv", error=str(exc))
            return
        self.writer.write("receiver_ready", source="fpv", bind=f"{self.bind_ip}:{self.bind_port}")
        while not self._stop.is_set():
            try:
                packet, address = sock.recvfrom(65536)
            except socket.timeout:
                self._discard_stale_frames(frames)
                continue
            except OSError:
                return
            if len(packet) < FRAME_HEADER.size:
                self.writer.write("frame_error", reason="short_packet", size=len(packet))
                continue
            values = FRAME_HEADER.unpack_from(packet)
            frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = values
            payload = packet[FRAME_HEADER.size:]
            if payload_size != len(payload):
                self.writer.write(
                    "frame_error",
                    reason="payload_size_mismatch",
                    frame_id=frame_id,
                    expected=payload_size,
                    actual=len(payload),
                )
                continue
            frame = frames.setdefault(
                frame_id,
                {
                    "chunks": {},
                    "total_chunks": total_chunks,
                    "jpeg_size": jpeg_size,
                    "sim_time_ns": sim_time_ns,
                    "first_seen_ns": time.monotonic_ns(),
                    "source": list(address),
                },
            )
            frame["chunks"][chunk_id] = payload
            if len(frame["chunks"]) == frame["total_chunks"]:
                self._save_frame(frame_id, frame)
                del frames[frame_id]
            self._discard_stale_frames(frames)

    def _save_frame(self, frame_id: int, frame: dict[str, Any]) -> None:
        try:
            jpeg = b"".join(
                frame["chunks"][index] for index in range(frame["total_chunks"])
            )
        except KeyError:
            return
        if len(jpeg) != frame["jpeg_size"]:
            self.writer.write(
                "frame_error",
                reason="jpeg_size_mismatch",
                frame_id=frame_id,
                expected=frame["jpeg_size"],
                actual=len(jpeg),
            )
            return
        filename = f"{frame_id:010d}_{frame['sim_time_ns']}.jpg"
        path = self.writer.frames_dir / filename
        path.write_bytes(jpeg)
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        fraction = 0.0
        if image is None:
            self.decode_failures += 1
        else:
            fraction = cyan_fraction(image)
            self.cyan_fraction_max = max(self.cyan_fraction_max, fraction)
            if fraction >= CYAN_PRESENT_FRACTION:
                self.frames_with_cyan += 1
        self.frame_count += 1
        self.writer.write(
            "frame",
            frame_id=frame_id,
            sim_time_ns=frame["sim_time_ns"],
            jpeg_size=len(jpeg),
            relative_path=str(Path("frames") / filename),
            width=int(image.shape[1]) if image is not None else None,
            height=int(image.shape[0]) if image is not None else None,
            cyan_fraction=fraction,
            cyan_present=fraction >= CYAN_PRESENT_FRACTION,
        )

    def _discard_stale_frames(self, frames: dict[int, dict[str, Any]]) -> None:
        cutoff = time.monotonic_ns() - 2_000_000_000
        for frame_id, frame in list(frames.items()):
            if frame["first_seen_ns"] < cutoff:
                self.writer.write(
                    "frame_error",
                    reason="incomplete_timeout",
                    frame_id=frame_id,
                    received_chunks=len(frame["chunks"]),
                    expected_chunks=frame["total_chunks"],
                )
                del frames[frame_id]


@dataclass(frozen=True)
class ManualControlConfig:
    hover_thrust: float = 0.42
    thrust_step: float = 0.03
    roll_rate: float = 0.08
    pitch_rate: float = 0.08
    forward_speed: float = 1.0
    lateral_speed: float = 1.0
    vertical_speed: float = 0.6
    motor_hover: float = 0.5
    motor_thrust_step: float = 0.08
    motor_tilt_step: float = 0.04


@dataclass(frozen=True)
class FlightCommand:
    mode: str
    values: tuple[float, ...]


def map_manual_keys(
    keys: Iterable[str], mode: str, config: ManualControlConfig
) -> FlightCommand:
    pressed = {key.lower() for key in keys}
    forward = float("w" in pressed) - float("s" in pressed)
    right = float("d" in pressed) - float("a" in pressed)
    up = float("q" in pressed) - float("e" in pressed)
    if mode == "attitude":
        return FlightCommand(
            mode,
            (
                right * config.roll_rate,
                -forward * config.pitch_rate,
                0.0,
                _clamp(config.hover_thrust + up * config.thrust_step, 0.0, 1.0),
            ),
        )
    if mode == "velocity":
        return FlightCommand(
            mode,
            (
                forward * config.forward_speed,
                right * config.lateral_speed,
                -up * config.vertical_speed,
            ),
        )
    if mode == "motor":
        base = config.motor_hover + up * config.motor_thrust_step
        roll = right * config.motor_tilt_step
        pitch = forward * config.motor_tilt_step
        motors = (
            base + roll - pitch,
            base - roll - pitch,
            base + roll + pitch,
            base - roll + pitch,
        )
        return FlightCommand(mode, tuple(_clamp(value, 0.0, 1.0) for value in motors))
    raise ValueError(f"unsupported control mode: {mode}")


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def update_throttle(
    current: float,
    keys: Iterable[str],
    elapsed_s: float,
    ramp_per_second: float,
    maximum: float,
) -> float:
    """Integrate Q/E into a persistent manual throttle setpoint."""
    pressed = {key.lower() for key in keys}
    direction = float("q" in pressed) - float("e" in pressed)
    elapsed_s = _clamp(elapsed_s, 0.0, 0.1)
    return _clamp(current + direction * ramp_per_second * elapsed_s, 0.0, maximum)


class CommandSender:
    def __init__(self, connection):
        self.connection = connection
        self.client_started_ms = int(time.time() * 1000)

    def arm(self) -> None:
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0,
        )

    def reset(self) -> None:
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            MAVLINK_CMD_SIM_RESET,
            0,
            0, 0, 0, 0, 0, 0, 0,
        )

    def send_timesync(self) -> int:
        client_time_ns = time.time_ns()
        # This field ordering matches the organizer's VQ2 starter client.
        self.connection.mav.timesync_send(client_time_ns, 0)
        return client_time_ns

    def send(self, command: FlightCommand) -> None:
        if not all(math.isfinite(value) for value in command.values):
            raise ValueError("control command contains a non-finite value")
        now_ms = int(time.time() * 1000) - self.client_started_ms
        if command.mode == "attitude":
            roll, pitch, yaw, thrust = command.values
            self.connection.mav.set_attitude_target_send(
                now_ms,
                self.connection.target_system,
                self.connection.target_component,
                RATES_ATTITUDE_MASK,
                [1, 0, 0, 0],
                roll,
                pitch,
                yaw,
                thrust,
            )
        elif command.mode == "velocity":
            vx, vy, vz = command.values
            self.connection.mav.set_position_target_local_ned_send(
                now_ms,
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                VELOCITY_POSITION_MASK,
                0.0, 0.0, 0.0,
                vx, vy, vz,
                0.0, 0.0, 0.0,
                0.0, 0.0,
            )
        elif command.mode == "motor":
            controls = [*command.values, 0.0, 0.0, 0.0, 0.0]
            self.connection.mav.set_actuator_control_target_send(
                time.time_ns() // 1000,
                0,
                self.connection.target_system,
                self.connection.target_component,
                controls,
            )
        else:
            raise ValueError(f"unsupported control mode: {command.mode}")


class KeyPoller:
    """Poll held keys; Win32 polling avoids global-hook elevation requirements."""

    VIRTUAL_KEYS = {**{key: ord(key.upper()) for key in "wasdqer"}, "escape": 0x1B}

    def __init__(self):
        self._windows = os.name == "nt"
        if self._windows:
            self._get_async_key_state = ctypes.windll.user32.GetAsyncKeyState
        else:
            import keyboard

            self._keyboard = keyboard

    def pressed(self) -> set[str]:
        if self._windows:
            return {
                name
                for name, code in self.VIRTUAL_KEYS.items()
                if self._get_async_key_state(code) & 0x8000
            }
        return {
            name for name in self.VIRTUAL_KEYS if self._keyboard.is_pressed(name)
        }


def create_run_dir(root: Path, prefix: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{prefix}_{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{prefix}_{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    (run_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
