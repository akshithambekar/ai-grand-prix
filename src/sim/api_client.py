"""Simulator client interface.

This is deliberately thin because the official Python API has not been
released. Keep simulator-specific code behind this boundary.
"""

from __future__ import annotations

from control.commands import PilotCommand
from sim.types import CameraFrame, Telemetry


class SimulatorClient:
    """Minimal interface required by the autonomy loop."""

    def connect(self) -> None:
        raise NotImplementedError

    def read_telemetry(self) -> Telemetry:
        raise NotImplementedError

    def read_frame(self) -> CameraFrame:
        raise NotImplementedError

    def send_command(self, command: PilotCommand) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

