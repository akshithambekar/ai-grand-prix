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

    def __init__(self, gamma=0.995):
        self.gamma = gamma
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
        if observation is not None:
            self._set_potential_state(data, observation)

    def compute(self, data, observation, termination_reason=None) -> RewardResult:
        components = {"step": -0.01}
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
            components["position_progress"] = float(np.clip(
                self._previous_distance_m - distance, -0.5, 0.5
            ))
            half_width = max(self._finite(active_gate.get("width_m")) or 0.0, 0.1) / 2.0
            half_height = max(self._finite(active_gate.get("height_m")) or 0.0, 0.1) / 2.0
            lateral = abs(self._finite(active_gate.get("lateral_m")) or 0.0) / half_width
            vertical = abs(self._finite(active_gate.get("vertical_m")) or 0.0) / half_height
            components["gate_alignment"] = -0.05 * float(np.clip(
                math.hypot(lateral, vertical), 0.0, 2.0
            ))

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

        self._previous_gate_index = gate_index
        self._previous_track_id = track_id if detected else None
        self._previous_potential = potential
        self._previous_physical_gate_id = physical_gate_id if physical_valid else None
        self._previous_distance_m = distance
        return RewardResult(
            total=float(sum(components.values())),
            components=components,
        )

    def _set_potential_state(self, data, observation):
        active_gate = self._mapping(data.get("active_gate_state"))
        if active_gate.get("valid"):
            self._previous_physical_gate_id = active_gate.get("gate_id")
            self._previous_distance_m = self._finite(active_gate.get("distance_m"))
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
    def _mapping(value):
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _finite(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None
