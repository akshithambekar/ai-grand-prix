"""Short-horizon state estimator skeleton."""

from __future__ import annotations

from perception.observations import GateObservation
from sim.types import Telemetry


class StateEstimator:
    def update(
        self,
        telemetry: Telemetry,
        gate: GateObservation | None,
    ) -> None:
        raise NotImplementedError

