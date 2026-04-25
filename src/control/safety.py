"""Safety monitor and conservative failsafe commands."""

from __future__ import annotations

from config.settings import ControlSettings
from control.commands import PilotCommand
from perception.observations import GateObservation
from sim.types import Telemetry


class SafetyMonitor:
    def __init__(self, settings: ControlSettings) -> None:
        self.settings = settings

    def should_failsafe(
        self,
        telemetry: Telemetry,
        gate: GateObservation | None,
    ) -> bool:
        if gate is None:
            return False
        return telemetry.timestamp - gate.timestamp > self.settings.max_perception_age_s

    def failsafe_command(self, timestamp: float) -> PilotCommand:
        return PilotCommand(
            timestamp=timestamp,
            throttle=self.settings.failsafe_throttle,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
        )

