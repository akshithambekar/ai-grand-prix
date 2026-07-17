"""Canonical vehicle and active-gate state derived from asynchronous telemetry."""

import math
import time
from collections.abc import Mapping, Sequence

import numpy as np
from pymavlink import mavutil


STATE_FRESHNESS_S = 0.5
LOCAL_NED_FRAME = mavutil.mavlink.MAV_FRAME_LOCAL_NED


def finite_vector(value, size):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def quaternion_normalize(value):
    quaternion = finite_vector(value, 4)
    if quaternion is None:
        return None
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm < 1e-9:
        return None
    return tuple(item / norm for item in quaternion)


def quaternion_from_euler(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return quaternion_normalize((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ))


def euler_from_quaternion(value):
    q = quaternion_normalize(value)
    if q is None:
        return None
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_term = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_term)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def rotate_vector(quaternion, vector):
    """Rotate a vector by a body-to-parent-frame quaternion."""
    q = quaternion_normalize(quaternion)
    v = finite_vector(vector, 3)
    if q is None or v is None:
        return None
    w, x, y, z = q
    qv = np.asarray((x, y, z), dtype=np.float64)
    vec = np.asarray(v, dtype=np.float64)
    rotated = vec + 2.0 * np.cross(qv, np.cross(qv, vec) + w * vec)
    return tuple(float(item) for item in rotated)


def inverse_rotate_vector(quaternion, vector):
    q = quaternion_normalize(quaternion)
    if q is None:
        return None
    return rotate_vector((q[0], -q[1], -q[2], -q[3]), vector)


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _age(snapshot, now):
    received = snapshot.get("received_at_s")
    try:
        return max(0.0, now - float(received))
    except (TypeError, ValueError):
        return math.inf


def _fresh(snapshot, now, freshness_s):
    return bool(snapshot) and _age(snapshot, now) <= freshness_s


def derive_vehicle_state(data, *, now=None, freshness_s=STATE_FRESHNESS_S):
    """Build and atomically publish one non-duplicative canonical state snapshot."""
    now = time.monotonic() if now is None else float(now)
    odometry = _mapping(data.get("odometry"))
    local = _mapping(data.get("local_position_ned"))
    attitude = _mapping(data.get("attitude"))
    imu = _mapping(data.get("imu"))

    odom_fresh = _fresh(odometry, now, freshness_s)
    odom_compatible = odometry.get("frame_id") == LOCAL_NED_FRAME
    odom_position = finite_vector(odometry.get("position_ned"), 3)
    odom_q = quaternion_normalize(odometry.get("quaternion_wxyz"))
    odom_velocity_raw = finite_vector(odometry.get("velocity_ned"), 3)
    child_frame = odometry.get("child_frame_id")
    if child_frame == LOCAL_NED_FRAME:
        odom_velocity = odom_velocity_raw
    elif child_frame == getattr(mavutil.mavlink, "MAV_FRAME_BODY_FRD", 12):
        odom_velocity = rotate_vector(odom_q, odom_velocity_raw)
    else:
        odom_velocity = None
    odom_rates = finite_vector(odometry.get("body_rates"), 3)

    local_fresh = _fresh(local, now, freshness_s)
    local_position = finite_vector(local.get("position_ned"), 3)
    local_velocity = finite_vector(local.get("velocity_ned"), 3)
    attitude_fresh = _fresh(attitude, now, freshness_s)
    attitude_euler = finite_vector(attitude.get("euler"), 3)
    attitude_rates = finite_vector(attitude.get("body_rates"), 3)
    imu_fresh = _fresh(imu, now, freshness_s)
    acceleration = finite_vector(
        (imu.get("xacc"), imu.get("yacc"), imu.get("zacc")), 3
    ) if imu_fresh else None

    if odom_fresh and odom_compatible and odom_position is not None and odom_velocity is not None:
        position, velocity, position_source = odom_position, odom_velocity, "odometry"
    elif local_fresh and local_position is not None and local_velocity is not None:
        position, velocity, position_source = local_position, local_velocity, "local_position_ned"
    else:
        position, velocity, position_source = None, None, None

    if odom_fresh and odom_compatible and odom_q is not None:
        quaternion = odom_q
        euler = euler_from_quaternion(quaternion)
        attitude_source = "odometry"
    elif attitude_fresh and attitude_euler is not None:
        euler = attitude_euler
        quaternion = quaternion_from_euler(*euler)
        attitude_source = "attitude"
    else:
        quaternion, euler, attitude_source = None, None, None

    if odom_fresh and odom_compatible and odom_rates is not None:
        rates, rates_source = odom_rates, "odometry"
    elif attitude_fresh and attitude_rates is not None:
        rates, rates_source = attitude_rates, "attitude"
    else:
        rates, rates_source = None, None

    valid = (
        position is not None and velocity is not None and quaternion is not None
        and rates is not None and acceleration is not None
    )
    snapshot = {
        "valid": valid,
        "position_ned": position,
        "velocity_ned": velocity,
        "quaternion_wxyz": quaternion,
        "euler": euler,
        "body_rates": rates,
        "acceleration_body": acceleration,
        "position_source": position_source,
        "attitude_source": attitude_source,
        "rates_source": rates_source,
        "odometry_age_s": _age(odometry, now),
        "local_position_age_s": _age(local, now),
        "attitude_age_s": _age(attitude, now),
        "imu_age_s": _age(imu, now),
        "odometry_reset_counter": odometry.get("reset_counter"),
        "derived_at_s": now,
    }
    data["vehicle_state"] = snapshot
    derive_active_gate_state(data)
    return snapshot


def derive_active_gate_state(data):
    """Publish body-relative geometry for the race status' active gate."""
    vehicle = _mapping(data.get("vehicle_state"))
    track = _mapping(data.get("track"))
    status = _mapping(data.get("race_status"))
    gates = _mapping(track.get("gates"))
    gate_index = status.get("active_gate_index")
    gate = gates.get(gate_index)
    if gate is None:
        try:
            gate = gates.get(int(gate_index))
        except (TypeError, ValueError):
            gate = None

    invalid = {
        "valid": False,
        "gate_id": gate_index,
        "relative_position_body": None,
        "gate_normal_body": None,
        "distance_m": None,
        "lateral_m": None,
        "vertical_m": None,
        "width_m": None,
        "height_m": None,
    }
    if not vehicle.get("valid") or not track.get("track_geometry_valid") or not isinstance(gate, Mapping):
        data["active_gate_state"] = invalid
        return invalid

    vehicle_position = finite_vector(vehicle.get("position_ned"), 3)
    vehicle_q = quaternion_normalize(vehicle.get("quaternion_wxyz"))
    gate_position = finite_vector(gate.get("position_ned"), 3)
    gate_q = quaternion_normalize(gate.get("quaternion_wxyz"))
    try:
        width, height = float(gate.get("width_m")), float(gate.get("height_m"))
    except (TypeError, ValueError):
        data["active_gate_state"] = invalid
        return invalid
    if None in (vehicle_position, vehicle_q, gate_position, gate_q) or width <= 0 or height <= 0:
        data["active_gate_state"] = invalid
        return invalid

    relative_ned = tuple(gate_position[i] - vehicle_position[i] for i in range(3))
    relative_body = inverse_rotate_vector(vehicle_q, relative_ned)
    normal_ned = rotate_vector(gate_q, (1.0, 0.0, 0.0))
    toward_vehicle = tuple(vehicle_position[i] - gate_position[i] for i in range(3))
    if sum(normal_ned[i] * toward_vehicle[i] for i in range(3)) < 0.0:
        normal_ned = tuple(-item for item in normal_ned)
    normal_body = inverse_rotate_vector(vehicle_q, normal_ned)
    distance = math.sqrt(sum(item * item for item in relative_ned))
    snapshot = {
        "valid": True,
        "gate_id": gate.get("gate_id", gate_index),
        "relative_position_ned": relative_ned,
        "relative_position_body": relative_body,
        "gate_normal_body": normal_body,
        "distance_m": distance,
        "lateral_m": relative_body[1],
        "vertical_m": relative_body[2],
        "width_m": width,
        "height_m": height,
    }
    data["active_gate_state"] = snapshot
    return snapshot
