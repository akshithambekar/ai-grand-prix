import math
from collections.abc import Mapping

import numpy as np


OBSERVATION_SIZE = 19
AREA_FLOOR = 1e-6
GYRO_SCALE_RAD_S = 5.0
ACCEL_SCALE_M_S2 = 20.0
MISSED_FRAME_SCALE = 10.0


class ObservationEncoder:
    """Convert asynchronous telemetry snapshots into a normalized policy vector."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._previous_geometry = None

    def encode(self, data: Mapping[str, object], previous_action) -> np.ndarray:
        gate = self._mapping(data.get("gate"))
        status = self._mapping(data.get("race_status"))
        imu = self._mapping(data.get("imu"))

        detected, cx, cy, log_area = self._geometry(gate)
        track_id = gate.get("track_id")
        gate_index = status.get("active_gate_index")
        current_geometry = None
        deltas = (0.0, 0.0, 0.0)

        if detected:
            current_geometry = (track_id, gate_index, cx, cy, log_area)
            previous = self._previous_geometry
            if previous is not None and previous[:2] == current_geometry[:2]:
                deltas = tuple(
                    float(np.clip(current_geometry[i] - previous[i], -1.0, 1.0))
                    for i in range(2, 5)
                )

        self._previous_geometry = current_geometry
        action = np.asarray(previous_action, dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all():
            action = np.zeros(4, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        observation = np.asarray([
            float(detected),
            cx,
            cy,
            log_area,
            *deltas,
            np.clip(self._number(gate.get("tracking_confidence")), 0.0, 1.0),
            np.clip(
                self._number(gate.get("tracking_missed_frames")) / MISSED_FRAME_SCALE,
                0.0,
                1.0,
            ),
            np.clip(self._number(imu.get("xgyro")) / GYRO_SCALE_RAD_S, -1.0, 1.0),
            np.clip(self._number(imu.get("ygyro")) / GYRO_SCALE_RAD_S, -1.0, 1.0),
            np.clip(self._number(imu.get("zgyro")) / GYRO_SCALE_RAD_S, -1.0, 1.0),
            np.clip(self._number(imu.get("xacc")) / ACCEL_SCALE_M_S2, -1.0, 1.0),
            np.clip(self._number(imu.get("yacc")) / ACCEL_SCALE_M_S2, -1.0, 1.0),
            np.clip(self._number(imu.get("zacc")) / ACCEL_SCALE_M_S2, -1.0, 1.0),
            *action,
        ], dtype=np.float32)
        if observation.shape != (OBSERVATION_SIZE,) or not np.isfinite(observation).all():
            raise ValueError("observation encoder produced invalid output")
        return observation

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
        log_area = 2.0 * (
            (math.log(area_fraction) - math.log(AREA_FLOOR))
            / (0.0 - math.log(AREA_FLOOR))
        ) - 1.0
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
