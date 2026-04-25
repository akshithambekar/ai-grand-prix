"""Core runtime data types shared across the autonomy stack."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class EulerAngles:
    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True)
class Quaternion:
    w: float
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Telemetry:
    timestamp: float
    position: Vec3 | None
    velocity: Vec3 | None
    orientation: Quaternion | EulerAngles
    angular_velocity: Vec3 | None = None
    linear_acceleration: Vec3 | None = None


@dataclass(frozen=True)
class CameraFrame:
    timestamp: float
    image: NDArray[np.uint8]
    frame_id: int

