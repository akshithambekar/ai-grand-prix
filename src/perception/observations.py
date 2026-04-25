"""Perception output types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class GateObservation:
    timestamp: float
    visible: bool
    confidence: float
    center_x: float | None
    center_y: float | None
    width: float | None
    height: float | None
    corners: tuple[Point2D, ...] | None = None
    source: Literal["gate", "guidance", "tracker"] = "gate"

