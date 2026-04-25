"""Telemetry source abstraction for live and replayed vehicle state."""

from __future__ import annotations

from sim.types import Telemetry


class TelemetrySource:
    def read_telemetry(self) -> Telemetry:
        raise NotImplementedError

