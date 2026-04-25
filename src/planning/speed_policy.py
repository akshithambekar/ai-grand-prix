"""Confidence-aware speed policy."""

from __future__ import annotations

from config.settings import ControlSettings
from perception.observations import GateObservation


class SpeedPolicy:
    def __init__(self, settings: ControlSettings) -> None:
        self.settings = settings

    def target_pitch(self, gate: GateObservation | None) -> float:
        if gate is None or gate.confidence < self.settings.min_command_confidence:
            return 0.0
        return self.settings.approach_pitch

