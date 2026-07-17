"""Telemetry readiness checks shared by live Gym and the preflight probe."""

import time

from controls.state import derive_vehicle_state


def telemetry_report(data):
    state = data.get("vehicle_state") or {}
    track = data.get("track") or {}
    active_gate = data.get("active_gate_state") or {}
    return {
        "vehicle_state_valid": bool(state.get("valid")),
        "position_source": state.get("position_source"),
        "attitude_source": state.get("attitude_source"),
        "rates_source": state.get("rates_source"),
        "odometry_age_s": state.get("odometry_age_s"),
        "local_position_age_s": state.get("local_position_age_s"),
        "attitude_age_s": state.get("attitude_age_s"),
        "imu_age_s": state.get("imu_age_s"),
        "track_geometry_valid": bool(track.get("track_geometry_valid")),
        "active_gate_geometry_valid": bool(active_gate.get("valid")),
        "track_gate_count": len(track.get("gates") or {}),
        "track_rejected_gates": track.get("rejected_gates", 0),
        "odometry_reset_counter": state.get("odometry_reset_counter"),
    }


def wait_for_vehicle_state(data, timeout_s=10.0, poll_s=0.02):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if derive_vehicle_state(data).get("valid"):
            return telemetry_report(data)
        time.sleep(poll_s)
    report = telemetry_report(data)
    raise TimeoutError(
        f"state_v2 telemetry readiness failed after {timeout_s:.1f}s: {report}"
    )
