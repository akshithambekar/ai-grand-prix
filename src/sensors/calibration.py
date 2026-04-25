"""Camera calibration data.

VQ1 may be forgiving, but wide-angle distortion will matter once we estimate
gate centers and apparent scale from image coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CameraCalibration:
    intrinsic_matrix: NDArray[np.float64] | None = None
    distortion_coefficients: NDArray[np.float64] | None = None
    camera_to_body: NDArray[np.float64] | None = None

