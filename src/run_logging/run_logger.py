"""Structured run logger skeleton."""

from __future__ import annotations

from control.commands import PilotCommand
from perception.observations import GateObservation
from sim.types import CameraFrame, Telemetry


class RunLogger:
    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir

    def start_run(self, metadata: dict) -> None:
        _ = metadata

    def log_frame(self, frame: CameraFrame) -> None:
        _ = frame

    def log_telemetry(self, telemetry: Telemetry) -> None:
        _ = telemetry

    def log_observation(self, observation: GateObservation | None) -> None:
        _ = observation

    def log_command(self, command: PilotCommand) -> None:
        _ = command

    def close(self) -> None:
        pass

