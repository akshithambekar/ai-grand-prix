"""Image preprocessing for VQ1 gate detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sim.types import CameraFrame


@dataclass(frozen=True)
class PreprocessedFrame:
    timestamp: float
    image: NDArray[np.uint8]
    frame_id: int
    scale: float


class ImagePreprocessor:
    def __init__(self, max_width: int = 1280) -> None:
        self.max_width = max_width

    def preprocess(self, frame: CameraFrame) -> PreprocessedFrame:
        """Return a frame suitable for fast detection.

        Downsampling is deferred until OpenCV or another image backend is added.
        """
        return PreprocessedFrame(
            timestamp=frame.timestamp,
            image=frame.image,
            frame_id=frame.frame_id,
            scale=1.0,
        )

