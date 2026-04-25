"""Temporal tracking for gate observations."""

from __future__ import annotations

from config.settings import PerceptionSettings
from perception.observations import GateObservation
from sim.types import Telemetry


class GateTracker:
    """Select and stabilize the best gate observation across frames."""

    def __init__(self, settings: PerceptionSettings) -> None:
        self.settings = settings
        self.last_observation: GateObservation | None = None

    def update(
        self,
        observations: list[GateObservation],
        telemetry: Telemetry,
    ) -> GateObservation | None:
        _ = telemetry
        if not observations:
            return self.last_observation

        best = max(observations, key=lambda observation: observation.confidence)
        if best.confidence < self.settings.min_gate_confidence:
            return self.last_observation

        self.last_observation = best
        return best

