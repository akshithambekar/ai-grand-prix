"""Conservative image-space visual servo controller."""

from __future__ import annotations

from config.settings import ControlSettings
from control.commands import PilotCommand
from perception.observations import GateObservation
from sim.types import Telemetry
from state.race_state import RaceState


class VisualServoController:
    def __init__(self, settings: ControlSettings) -> None:
        self.settings = settings

    def compute_command(
        self,
        telemetry: Telemetry,
        gate: GateObservation,
        race_state: RaceState,
    ) -> PilotCommand:
        _ = race_state
        horizontal_error = (gate.center_x or 0.5) - 0.5
        vertical_error = 0.5 - (gate.center_y or 0.5)

        command = PilotCommand(
            timestamp=telemetry.timestamp,
            throttle=self.settings.hover_throttle + self.settings.vertical_gain * vertical_error,
            roll=self.settings.roll_gain * horizontal_error,
            pitch=self.settings.approach_pitch if gate.confidence >= self.settings.min_command_confidence else 0.0,
            yaw=self.settings.yaw_gain * horizontal_error,
        )
        return command.clamped(self.settings.limits)

