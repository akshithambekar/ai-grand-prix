"""Visual guidance-aid detector skeleton for VQ1."""

from __future__ import annotations

from perception.observations import GateObservation
from sim.types import CameraFrame


class GuidanceDetector:
    def detect(self, frame: CameraFrame) -> list[GateObservation]:
        raise NotImplementedError

