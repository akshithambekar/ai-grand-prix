"""Frame source abstraction for live and replayed camera data."""

from __future__ import annotations

from sim.types import CameraFrame


class FrameSource:
    def read_frame(self) -> CameraFrame:
        raise NotImplementedError

