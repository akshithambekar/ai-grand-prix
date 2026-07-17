import math
from collections.abc import Mapping

import numpy as np

from controls.state import derive_active_gate_state, derive_vehicle_state, finite_vector


OBSERVATION_SCHEMA = "state_v3"
OBSERVATION_FIELDS = (
    "vision_detected", "vision_center_x", "vision_center_y", "vision_log_area",
    "tracking_confidence", "tracking_missed_frames",
    "vehicle_state_valid",
    "position_n", "position_e", "position_d",
    "velocity_n", "velocity_e", "velocity_d",
    "sin_roll", "cos_roll", "sin_pitch", "cos_pitch", "sin_yaw", "cos_yaw",
    "body_rate_roll", "body_rate_pitch", "body_rate_yaw",
    "acceleration_x", "acceleration_y", "acceleration_z",
    "track_geometry_valid",
    "gate_plane_distance", "gate_lateral", "gate_vertical",
    "gate_normal_body_x", "gate_normal_body_y", "gate_normal_body_z",
    "gate_width", "gate_height",
    "previous_roll", "previous_pitch", "previous_yaw", "previous_thrust",
)
OBSERVATION_SIZE = len(OBSERVATION_FIELDS)
AREA_FLOOR = 1e-6
POSITION_SCALE_M = 100.0
VELOCITY_SCALE_M_S = 20.0
BODY_RATE_SCALE_RAD_S = 5.0
ACCEL_SCALE_M_S2 = 20.0
GATE_POSITION_SCALE_M = 50.0
GATE_DIMENSION_SCALE_M = 10.0
MISSED_FRAME_SCALE = 10.0


class ObservationEncoder:
    """Convert asynchronous telemetry snapshots into the state_v3 policy vector."""

    def reset(self):
        pass

    def encode(self, data: Mapping[str, object], previous_action) -> np.ndarray:
        if any(key in data for key in ("odometry", "local_position_ned", "attitude")):
            derive_vehicle_state(data)
        elif "active_gate_state" not in data:
            derive_active_gate_state(data)

        gate = self._mapping(data.get("gate"))
        vehicle = self._mapping(data.get("vehicle_state"))
        active_gate = self._mapping(data.get("active_gate_state"))
        detected, cx, cy, log_area = self._geometry(gate)
        vehicle_valid = bool(vehicle.get("valid"))
        track_valid = bool(active_gate.get("valid"))

        position = self._vector(vehicle.get("position_ned"), vehicle_valid)
        velocity = self._vector(vehicle.get("velocity_ned"), vehicle_valid)
        euler = self._vector(vehicle.get("euler"), vehicle_valid)
        rates = self._vector(vehicle.get("body_rates"), vehicle_valid)
        acceleration = self._vector(vehicle.get("acceleration_body"), vehicle_valid)
        relative = self._vector(active_gate.get("relative_position_gate"), track_valid)
        if track_valid:
            relative = (abs(relative[0]), relative[1], relative[2])
        normal = self._vector(active_gate.get("gate_normal_body"), track_valid)

        action = np.asarray(previous_action, dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all():
            action = np.zeros(4, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        attitude_features = (
            math.sin(euler[0]), math.cos(euler[0]),
            math.sin(euler[1]), math.cos(euler[1]),
            math.sin(euler[2]), math.cos(euler[2]),
        ) if vehicle_valid else (0.0,) * 6
        dimensions = (
            self._number(active_gate.get("width_m")) / GATE_DIMENSION_SCALE_M,
            self._number(active_gate.get("height_m")) / GATE_DIMENSION_SCALE_M,
        ) if track_valid else (0.0, 0.0)

        observation = np.asarray([
            float(detected), cx, cy, log_area,
            np.clip(self._number(gate.get("tracking_confidence")), 0.0, 1.0),
            np.clip(self._number(gate.get("tracking_missed_frames")) / MISSED_FRAME_SCALE, 0.0, 1.0),
            float(vehicle_valid),
            *(item / POSITION_SCALE_M for item in position),
            *(item / VELOCITY_SCALE_M_S for item in velocity),
            *attitude_features,
            *(item / BODY_RATE_SCALE_RAD_S for item in rates),
            *(item / ACCEL_SCALE_M_S2 for item in acceleration),
            float(track_valid),
            *(item / GATE_POSITION_SCALE_M for item in relative),
            *normal,
            *dimensions,
            *action,
        ], dtype=np.float32)
        observation = np.clip(observation, -1.0, 1.0)
        if observation.shape != (OBSERVATION_SIZE,) or not np.isfinite(observation).all():
            raise ValueError("state_v3 observation encoder produced invalid output")
        return observation

    @staticmethod
    def _vector(value, valid):
        vector = finite_vector(value, 3) if valid else None
        return vector if vector is not None else (0.0, 0.0, 0.0)

    @staticmethod
    def _geometry(gate):
        if not gate.get("detected", False):
            return False, 0.0, 0.0, 0.0
        centroid = gate.get("centroid")
        frame_size = gate.get("frame_size")
        area = gate.get("area_px")
        if not centroid or not frame_size or area is None:
            return False, 0.0, 0.0, 0.0
        width, height = frame_size
        if width <= 0 or height <= 0:
            return False, 0.0, 0.0, 0.0
        cx = float(np.clip(2.0 * float(centroid[0]) / width - 1.0, -1.0, 1.0))
        cy = float(np.clip(2.0 * float(centroid[1]) / height - 1.0, -1.0, 1.0))
        area_fraction = float(np.clip(float(area) / (width * height), AREA_FLOOR, 1.0))
        log_area = 2.0 * ((math.log(area_fraction) - math.log(AREA_FLOOR)) / -math.log(AREA_FLOOR)) - 1.0
        return True, cx, cy, float(np.clip(log_area, -1.0, 1.0))

    @staticmethod
    def _mapping(value):
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _number(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return number if math.isfinite(number) else 0.0
