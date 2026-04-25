"""Search behavior for reacquiring gates."""

from __future__ import annotations

from config.settings import ControlSettings
from control.commands import PilotCommand
from sim.types import Telemetry
from state.race_state import RaceState


class SearchBehavior:
    def __init__(self, settings: ControlSettings) -> None:
        self.settings = settings

    def command(self, telemetry: Telemetry, race_state: RaceState) -> PilotCommand:
        _ = race_state
        return PilotCommand(
            timestamp=telemetry.timestamp,
            throttle=self.settings.search_throttle,
            roll=0.0,
            pitch=0.0,
            yaw=self.settings.search_yaw,
        )

