"""Command smoothing to avoid abrupt control changes."""

from __future__ import annotations

from config.settings import ControlSettings
from control.commands import PilotCommand


class CommandSmoother:
    def __init__(self, settings: ControlSettings) -> None:
        self.alpha = settings.command_smoothing_alpha

    def smooth(
        self,
        previous: PilotCommand | None,
        desired: PilotCommand,
    ) -> PilotCommand:
        if previous is None:
            return desired

        return PilotCommand(
            timestamp=desired.timestamp,
            throttle=self._mix(previous.throttle, desired.throttle),
            roll=self._mix(previous.roll, desired.roll),
            pitch=self._mix(previous.pitch, desired.pitch),
            yaw=self._mix(previous.yaw, desired.yaw),
        )

    def _mix(self, previous: float, desired: float) -> float:
        return previous * (1.0 - self.alpha) + desired * self.alpha

