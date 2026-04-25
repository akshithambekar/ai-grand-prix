"""Offline replay skeleton for recorded simulator runs."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from perception.observations import GateObservation
from sim.types import CameraFrame, Telemetry


@dataclass(frozen=True)
class ReplaySample:
    """Synchronized data from a recorded run."""

    frame: CameraFrame
    telemetry: Telemetry
    gate: GateObservation | None = None


class LogReplay:
    """Read recorded run data for offline perception and control tuning."""

    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir

    def synchronized_samples(self) -> Iterator[ReplaySample]:
        """Yield synchronized replay samples.

        The storage format is intentionally deferred until we see the simulator
        frame format and logging constraints.
        """
        raise NotImplementedError

