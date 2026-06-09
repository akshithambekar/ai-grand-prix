"""Thread-safe telemetry cache and MAVLink ingest for simulator perception."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import struct
import threading
import time
from typing import Any

from perception.types import GatePoseWorld, TelemetrySnapshot

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID = 2
_TRACK_GATE_STRUCT = struct.Struct("<Hfffffffff")
_RACE_STATUS_STRUCT = struct.Struct("<BQqqIq")


@dataclass(frozen=True, slots=True)
class RaceStatus:
    """Decoded race-status payload from the simulator."""

    sim_boot_time_ms: int
    race_start_boot_time_ms: int | None
    race_finish_time_ns: int | None
    active_gate_index: int
    last_gate_race_time: float | None


def parse_race_status_payload(payload: bytes) -> RaceStatus:
    """Decode simulator race status from an encapsulated MAVLink payload."""

    if len(payload) < _RACE_STATUS_STRUCT.size:
        msg = (
            f"race status payload too short: expected at least {_RACE_STATUS_STRUCT.size} bytes, "
            f"got {len(payload)}"
        )
        raise ValueError(msg)

    (
        _data_type,
        sim_boot_time_ms,
        race_start_boot_time_ms,
        race_finish_time_ns,
        active_gate_index,
        last_gate_race_time,
    ) = _RACE_STATUS_STRUCT.unpack_from(payload)

    return RaceStatus(
        sim_boot_time_ms=sim_boot_time_ms,
        race_start_boot_time_ms=race_start_boot_time_ms
        if race_start_boot_time_ms >= 0
        else None,
        race_finish_time_ns=race_finish_time_ns if race_finish_time_ns >= 0 else None,
        active_gate_index=active_gate_index,
        last_gate_race_time=float(last_gate_race_time) if last_gate_race_time >= 0 else None,
    )


def parse_track_data_payload(payload: bytes) -> dict[int, GatePoseWorld]:
    """Decode simulator gate geometry from the track-info payload."""

    if len(payload) < 2:
        msg = "track data payload too short to contain gate count"
        raise ValueError(msg)

    (num_gates,) = struct.unpack_from("<H", payload)
    offset = 2
    gate_map: dict[int, GatePoseWorld] = {}

    for _ in range(num_gates):
        if len(payload) < offset + _TRACK_GATE_STRUCT.size:
            msg = f"track data payload truncated while decoding gate {len(gate_map)}"
            raise ValueError(msg)

        (
            gate_id,
            position_ned_x,
            position_ned_y,
            position_ned_z,
            orientation_ned_w,
            orientation_ned_x,
            orientation_ned_y,
            orientation_ned_z,
            width,
            height,
        ) = _TRACK_GATE_STRUCT.unpack_from(payload, offset)
        offset += _TRACK_GATE_STRUCT.size

        gate_map[gate_id] = GatePoseWorld(
            gate_id=gate_id,
            center_position_m=(position_ned_x, position_ned_y, position_ned_z),
            orientation_xyzw=(
                orientation_ned_x,
                orientation_ned_y,
                orientation_ned_z,
                orientation_ned_w,
            ),
            width_m=width,
            height_m=height,
        )

    return gate_map


class TelemetryStateCache:
    """Thread-safe cache of the latest simulator telemetry and gate map."""

    def __init__(self, history_size: int = 512) -> None:
        self._history: deque[TelemetrySnapshot] = deque(maxlen=history_size)
        self._lock = threading.Lock()

        self._timestamp_ns: int | None = None
        self._vehicle_position_m: tuple[float, float, float] | None = None
        self._vehicle_orientation_xyzw: tuple[float, float, float, float] | None = None
        self._velocity_mps: tuple[float, float, float] | None = None
        self._body_rates_rad_s: tuple[float, float, float] | None = None
        self._motor_outputs: tuple[float, float, float, float] | None = None
        self._active_gate_index: int | None = None
        self._gate_map: dict[int, GatePoseWorld] = {}
        self._sim_boot_time_ms: int | None = None
        self._race_start_boot_time_ms: int | None = None
        self._race_finish_time_ns: int | None = None
        self._last_gate_race_time: float | None = None

    def update_pose(
        self,
        *,
        timestamp_ns: int | None,
        position_m: tuple[float, float, float] | None = None,
        orientation_xyzw: tuple[float, float, float, float] | None = None,
        velocity_mps: tuple[float, float, float] | None = None,
        body_rates_rad_s: tuple[float, float, float] | None = None,
    ) -> TelemetrySnapshot:
        with self._lock:
            if position_m is not None:
                self._vehicle_position_m = position_m
            if orientation_xyzw is not None:
                self._vehicle_orientation_xyzw = orientation_xyzw
            if velocity_mps is not None:
                self._velocity_mps = velocity_mps
            if body_rates_rad_s is not None:
                self._body_rates_rad_s = body_rates_rad_s
            return self._commit_locked(timestamp_ns)

    def update_body_rates(
        self, *, timestamp_ns: int | None, body_rates_rad_s: tuple[float, float, float]
    ) -> TelemetrySnapshot:
        with self._lock:
            self._body_rates_rad_s = body_rates_rad_s
            return self._commit_locked(timestamp_ns)

    def update_motor_outputs(
        self, *, timestamp_ns: int | None, motor_outputs: tuple[float, float, float, float]
    ) -> TelemetrySnapshot:
        with self._lock:
            self._motor_outputs = motor_outputs
            return self._commit_locked(timestamp_ns)

    def update_race_status(
        self,
        *,
        timestamp_ns: int | None,
        sim_boot_time_ms: int,
        race_start_boot_time_ms: int | None,
        race_finish_time_ns: int | None,
        active_gate_index: int,
        last_gate_race_time: float | None,
    ) -> TelemetrySnapshot:
        with self._lock:
            self._sim_boot_time_ms = sim_boot_time_ms
            self._race_start_boot_time_ms = race_start_boot_time_ms
            self._race_finish_time_ns = race_finish_time_ns
            self._active_gate_index = active_gate_index
            self._last_gate_race_time = last_gate_race_time
            return self._commit_locked(timestamp_ns)

    def update_gate_map(
        self, *, timestamp_ns: int | None, gate_map: dict[int, GatePoseWorld]
    ) -> TelemetrySnapshot:
        with self._lock:
            self._gate_map = dict(gate_map)
            return self._commit_locked(timestamp_ns)

    def get_latest_snapshot(self) -> TelemetrySnapshot | None:
        with self._lock:
            if not self._history:
                return None
            return self._history[-1]

    def get_snapshot_near(
        self, *, target_timestamp_ns: int | None, max_age_ns: int
    ) -> tuple[TelemetrySnapshot | None, int | None]:
        with self._lock:
            if not self._history:
                return None, None

            if target_timestamp_ns is None:
                latest = self._history[-1]
                return latest, 0

            candidate = next(
                (snapshot for snapshot in reversed(self._history) if snapshot.timestamp_ns <= target_timestamp_ns),
                self._history[-1],
            )
            age_ns = abs(target_timestamp_ns - candidate.timestamp_ns)
            if age_ns > max_age_ns:
                return None, age_ns
            return candidate, age_ns

    def _commit_locked(self, timestamp_ns: int | None) -> TelemetrySnapshot:
        effective_timestamp_ns = self._resolve_timestamp_locked(timestamp_ns)
        snapshot = TelemetrySnapshot(
            timestamp_ns=effective_timestamp_ns,
            vehicle_position_m=self._vehicle_position_m,
            vehicle_orientation_xyzw=self._vehicle_orientation_xyzw,
            velocity_mps=self._velocity_mps,
            body_rates_rad_s=self._body_rates_rad_s,
            motor_outputs=self._motor_outputs,
            active_gate_index=self._active_gate_index,
            gate_map=self._gate_map,
            sim_boot_time_ms=self._sim_boot_time_ms,
            race_start_boot_time_ms=self._race_start_boot_time_ms,
            race_finish_time_ns=self._race_finish_time_ns,
            last_gate_race_time=self._last_gate_race_time,
        )
        self._timestamp_ns = effective_timestamp_ns
        self._history.append(snapshot)
        return snapshot

    def _resolve_timestamp_locked(self, timestamp_ns: int | None) -> int:
        if timestamp_ns is not None:
            return timestamp_ns
        if self._timestamp_ns is not None:
            return self._timestamp_ns
        return time.time_ns()


class MAVLinkTelemetryReceiver:
    """Background MAVLink ingest that continuously updates a telemetry cache."""

    def __init__(
        self,
        mavlink_connection: Any,
        cache: TelemetryStateCache,
        *,
        poll_interval_s: float = 0.001,
    ) -> None:
        self._mavlink_connection = mavlink_connection
        self._cache = cache
        self._poll_interval_s = poll_interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._track_chunks: dict[int, dict[int, bytes]] = {}
        self._expected_track_chunks: dict[int, int] = {}

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._receive_loop, name="mavlink-telemetry")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _receive_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                message = self._mavlink_connection.recv_match(blocking=False)
            except ConnectionResetError:
                return

            if message is None:
                time.sleep(self._poll_interval_s)
                continue

            self.on_message(message)

    def on_message(self, message: Any) -> None:
        message_type = message.get_type()
        if message_type == "BAD_DATA":
            return

        if message_type == "ATTITUDE":
            self._cache.update_pose(
                timestamp_ns=_coerce_message_timestamp_ns(message),
                orientation_xyzw=_euler_to_quaternion_xyzw(message.roll, message.pitch, message.yaw),
                body_rates_rad_s=(message.rollspeed, message.pitchspeed, message.yawspeed),
            )
            return

        if message_type == "LOCAL_POSITION_NED":
            self._cache.update_pose(
                timestamp_ns=_coerce_message_timestamp_ns(message),
                position_m=(message.x, message.y, message.z),
                velocity_mps=(message.vx, message.vy, message.vz),
            )
            return

        if message_type == "ODOMETRY":
            self._cache.update_pose(
                timestamp_ns=_coerce_message_timestamp_ns(message),
                position_m=(message.x, message.y, message.z),
                orientation_xyzw=(message.q[1], message.q[2], message.q[3], message.q[0]),
                velocity_mps=(message.vx, message.vy, message.vz),
                body_rates_rad_s=(message.rollspeed, message.pitchspeed, message.yawspeed),
            )
            return

        if message_type == "HIGHRES_IMU":
            self._cache.update_body_rates(
                timestamp_ns=_coerce_message_timestamp_ns(message),
                body_rates_rad_s=(message.xgyro, message.ygyro, message.zgyro),
            )
            return

        if message_type == "ACTUATOR_OUTPUT_STATUS":
            self._cache.update_motor_outputs(
                timestamp_ns=_coerce_message_timestamp_ns(message),
                motor_outputs=tuple(float(value) for value in message.actuator[:4]),
            )
            return

        if message_type == "DATA_TRANSMISSION_HANDSHAKE":
            transfer_id = message.width
            self._track_chunks[transfer_id] = {}
            self._expected_track_chunks[transfer_id] = message.packets
            return

        if message_type != "ENCAPSULATED_DATA":
            return

        raw_payload = bytes(message.data)
        if not raw_payload:
            return

        payload_type = raw_payload[0]
        if payload_type == ENCAPSULATED_RACE_STATUS_MSG_ID:
            race_status = parse_race_status_payload(raw_payload)
            self._cache.update_race_status(
                timestamp_ns=race_status.sim_boot_time_ms * 1_000_000,
                sim_boot_time_ms=race_status.sim_boot_time_ms,
                race_start_boot_time_ms=race_status.race_start_boot_time_ms,
                race_finish_time_ns=race_status.race_finish_time_ns,
                active_gate_index=race_status.active_gate_index,
                last_gate_race_time=race_status.last_gate_race_time,
            )
            return

        if payload_type != ENCAPSULATED_TRACK_INFO_MSG_ID:
            return

        self._on_track_data_packet(message, raw_payload)

    def _on_track_data_packet(self, message: Any, payload: bytes) -> None:
        _data_type, transfer_id = struct.unpack_from("<BH", payload)
        if transfer_id not in self._expected_track_chunks:
            return

        self._track_chunks[transfer_id][message.seqnr] = payload[3:]
        expected_chunks = self._expected_track_chunks[transfer_id]
        if len(self._track_chunks[transfer_id]) != expected_chunks:
            return

        full_payload = b"".join(
            self._track_chunks[transfer_id][index] for index in range(expected_chunks)
        )
        del self._track_chunks[transfer_id]
        del self._expected_track_chunks[transfer_id]
        gate_map = parse_track_data_payload(full_payload)
        self._cache.update_gate_map(timestamp_ns=None, gate_map=gate_map)


def _coerce_message_timestamp_ns(message: Any) -> int | None:
    time_usec = getattr(message, "time_usec", None)
    if time_usec is not None and time_usec >= 0:
        return int(time_usec) * 1_000

    time_boot_ms = getattr(message, "time_boot_ms", None)
    if time_boot_ms is not None and time_boot_ms >= 0:
        return int(time_boot_ms) * 1_000_000

    return None


def _euler_to_quaternion_xyzw(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5

    cos_r = math.cos(half_roll)
    sin_r = math.sin(half_roll)
    cos_p = math.cos(half_pitch)
    sin_p = math.sin(half_pitch)
    cos_y = math.cos(half_yaw)
    sin_y = math.sin(half_yaw)

    x = sin_r * cos_p * cos_y - cos_r * sin_p * sin_y
    y = cos_r * sin_p * cos_y + sin_r * cos_p * sin_y
    z = cos_r * cos_p * sin_y - sin_r * sin_p * cos_y
    w = cos_r * cos_p * cos_y + sin_r * sin_p * sin_y
    return (x, y, z, w)
