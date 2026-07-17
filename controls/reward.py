import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardResult:
    total: float
    components: Mapping[str, float]


class RewardCalculator:
    """Reward gate passage while guarding temporal terms across target changes."""

    def __init__(
        self,
        gamma=0.995,
        rate_slew_weight=0.02,
        thrust_slew_weight=0.005,
        alignment_weight=0.20,
        alignment_progress_weight=0.25,
    ):
        self.gamma = gamma
        self.rate_slew_weight = float(rate_slew_weight)
        self.thrust_slew_weight = float(thrust_slew_weight)
        self.alignment_weight = float(alignment_weight)
        self.alignment_progress_weight = float(alignment_progress_weight)
        if min(
            self.rate_slew_weight,
            self.thrust_slew_weight,
            self.alignment_weight,
            self.alignment_progress_weight,
        ) < 0.0:
            raise ValueError("reward weights must be non-negative")
        self.reset()

    def reset(self, data=None, observation=None):
        data = data or {}
        status = self._mapping(data.get("race_status"))
        collision = self._mapping(data.get("collision"))
        self._previous_gate_index = status.get("active_gate_index")
        self._collision_sequence = int(collision.get("sequence", 0) or 0)
        self._previous_track_id = None
        self._previous_potential = None
        self._previous_physical_gate_id = None
        self._previous_distance_m = None
        self._previous_alignment_error = None
        self._previous_action = self._action(observation)
        self._reference_down_m = self._vehicle_down(data)
        if observation is not None:
            self._set_potential_state(data, observation)

    def compute(self, data, observation, termination_reason=None) -> RewardResult:
        components = {"step": -0.01}
        current_action = self._action(observation)
        action_delta = current_action - self._previous_action
        components["rate_slew"] = -self.rate_slew_weight * float(
            np.dot(action_delta[:3], action_delta[:3])
        )
        components["thrust_slew"] = -self.thrust_slew_weight * float(
            action_delta[3] * action_delta[3]
        )
        status = self._mapping(data.get("race_status"))
        gate = self._mapping(data.get("gate"))
        gate_index = status.get("active_gate_index")

        increment = 0
        if gate_index is not None and self._previous_gate_index is not None:
            increment = max(0, int(gate_index) - int(self._previous_gate_index))
        if increment:
            components["gate_pass"] = 20.0 * increment

        active_gate = self._mapping(data.get("active_gate_state"))
        physical_valid = bool(active_gate.get("valid"))
        physical_gate_id = active_gate.get("gate_id")
        distance = self._finite(active_gate.get("distance_m")) if physical_valid else None
        same_physical_target = (
            distance is not None
            and self._previous_distance_m is not None
            and physical_gate_id == self._previous_physical_gate_id
            and gate_index == self._previous_gate_index
        )
        if same_physical_target:
            raw_progress = float(np.clip(
                self._previous_distance_m - distance, -0.5, 0.5
            ))
            alignment_error = self._physical_alignment_error(active_gate)
            if alignment_error is not None:
                # Approaching quickly only pays when the vehicle is lined up with the
                # opening. Moving away remains fully negative at every alignment.
                alignment_quality = math.exp(-3.0 * alignment_error * alignment_error)
                components["position_progress"] = (
                    raw_progress * alignment_quality if raw_progress > 0.0 else raw_progress
                )
                components["gate_alignment"] = -self.alignment_weight * float(
                    np.clip(alignment_error, 0.0, 2.0)
                )
                if self._previous_alignment_error is not None:
                    components["alignment_progress"] = self.alignment_progress_weight * float(
                        np.clip(self._previous_alignment_error - alignment_error, -0.5, 0.5)
                    )
            else:
                components["position_progress"] = raw_progress

        vehicle = self._mapping(data.get("vehicle_state"))
        if vehicle.get("valid"):
            velocity = self._vector(vehicle.get("velocity_ned"), 3)
            euler = self._vector(vehicle.get("euler"), 3)
            position = self._vector(vehicle.get("position_ned"), 3)
            if velocity is not None:
                # LOCAL_NED uses positive-down velocity. Only unsafe descent is
                # penalized; horizontal/total speed never appears in the reward.
                descent_rate = max(0.0, velocity[2] - 0.25)
                components["descent_stability"] = -0.05 * float(
                    np.clip(descent_rate, 0.0, 3.0)
                )
            if euler is not None:
                attitude_error = (euler[0] / 0.5) ** 2 + (euler[1] / 0.5) ** 2
                components["attitude_stability"] = -0.05 * float(
                    np.clip(attitude_error, 0.0, 2.0)
                )
            if position is not None and self._reference_down_m is not None:
                altitude_loss = max(0.0, position[2] - self._reference_down_m - 0.25)
                components["altitude_stability"] = -0.10 * float(
                    np.clip(altitude_loss / 2.0, 0.0, 1.0)
                )

        detected = bool(observation[0] > 0.5)
        track_id = gate.get("track_id")
        potential = self._potential(observation) if detected else None
        same_target = (
            potential is not None
            and self._previous_potential is not None
            and track_id == self._previous_track_id
            and gate_index == self._previous_gate_index
        )
        if same_target and not physical_valid:
            dense = self.gamma * potential - self._previous_potential
            components["visual_progress"] = float(np.clip(dense, -0.5, 0.5))

        collision = self._mapping(data.get("collision"))
        sequence = int(collision.get("sequence", 0) or 0)
        if sequence > self._collision_sequence:
            collision_id = collision.get("collision_id")
            threat = int(collision.get("threat_level", 0) or 0)
            if collision_id == 1002 or (collision_id == 1001 and threat >= 2):
                components["collision"] = -20.0
            elif collision_id == 1001:
                components["gate_contact"] = -2.0
            self._collision_sequence = sequence

        if termination_reason == "course_complete":
            components["course_complete"] = 50.0
        elif termination_reason in {
            "inverted", "tumbling", "position_divergence", "out_of_bounds", "stuck",
        }:
            components["terminal_failure"] = -20.0
        elif termination_reason in {"gate_timeout", "episode_timeout", "gate_lost"}:
            components["terminal_failure"] = -10.0

        self._previous_gate_index = gate_index
        self._previous_track_id = track_id if detected else None
        self._previous_potential = potential
        self._previous_physical_gate_id = physical_gate_id if physical_valid else None
        self._previous_distance_m = distance
        self._previous_alignment_error = (
            self._physical_alignment_error(active_gate) if physical_valid else None
        )
        self._previous_action = current_action
        return RewardResult(
            total=float(sum(components.values())),
            components=components,
        )

    def _set_potential_state(self, data, observation):
        active_gate = self._mapping(data.get("active_gate_state"))
        if active_gate.get("valid"):
            self._previous_physical_gate_id = active_gate.get("gate_id")
            self._previous_distance_m = self._finite(active_gate.get("distance_m"))
            self._previous_alignment_error = self._physical_alignment_error(active_gate)
        gate = self._mapping(data.get("gate"))
        if observation[0] <= 0.5:
            return
        self._previous_track_id = gate.get("track_id")
        self._previous_potential = self._potential(observation)

    @staticmethod
    def _potential(observation):
        cx, cy, log_area = (float(observation[i]) for i in (1, 2, 3))
        error = math.sqrt(cx * cx + cy * cy)
        potential = -error
        if error < 0.4:
            potential += 0.15 * log_area
        return potential

    @staticmethod
    def _action(observation):
        if observation is None:
            return np.zeros(4, dtype=np.float32)
        action = np.asarray(observation[-4:], dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all():
            return np.zeros(4, dtype=np.float32)
        return np.clip(action, -1.0, 1.0)

    @classmethod
    def _physical_alignment_error(cls, active_gate):
        width = cls._finite(active_gate.get("width_m"))
        height = cls._finite(active_gate.get("height_m"))
        lateral = cls._finite(active_gate.get("lateral_m"))
        vertical = cls._finite(active_gate.get("vertical_m"))
        if None in (width, height, lateral, vertical) or width <= 0.0 or height <= 0.0:
            return None
        return math.hypot(abs(lateral) / max(width / 2.0, 0.1), abs(vertical) / max(height / 2.0, 0.1))

    @classmethod
    def _vehicle_down(cls, data):
        vehicle = cls._mapping(data.get("vehicle_state"))
        position = cls._vector(vehicle.get("position_ned"), 3) if vehicle.get("valid") else None
        return position[2] if position is not None else None

    @staticmethod
    def _vector(value, size):
        try:
            vector = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return None
        if len(vector) != size or not all(math.isfinite(item) for item in vector):
            return None
        return vector

    @staticmethod
    def _mapping(value):
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _finite(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None
