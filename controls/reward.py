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

        detected = bool(observation[0] > 0.5)
        track_id = gate.get("track_id")
        potential = self._potential(observation) if detected else None
        same_target = (
            potential is not None
            and self._previous_potential is not None
            and track_id == self._previous_track_id
            and gate_index == self._previous_gate_index
        )
        if same_target:
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
        return RewardResult(
            total=float(sum(components.values())),
            components=components,
        )

    def _set_potential_state(self, data, observation):
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
