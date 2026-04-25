"""VQ1 gate detector skeleton."""

from __future__ import annotations

from config.settings import PerceptionSettings
from perception.image_preprocess import ImagePreprocessor
from perception.observations import GateObservation
from sim.types import CameraFrame


class GateDetector:
    """Detect highlighted VQ1 gates from FPV frames."""

    def __init__(self, settings: PerceptionSettings) -> None:
        self.settings = settings
        self.preprocessor = ImagePreprocessor(settings.max_processing_width)

    def detect(self, frame: CameraFrame) -> list[GateObservation]:
        """Return candidate gate observations sorted by confidence."""
        _ = self.preprocessor.preprocess(frame)
        return []

