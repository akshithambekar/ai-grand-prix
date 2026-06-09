"""Typed data contracts for the simulator-first perception stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

Point2D = tuple[float, float]
Point3D = tuple[float, float, float]
QuaternionXYZW = tuple[float, float, float, float]
BBoxXYXY = tuple[int, int, int, int]
BBoxFloatXYXY = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class GatePoseWorld:
    """Static world-space gate geometry keyed by gate identifier."""

    gate_id: int
    center_position_m: Point3D
    orientation_xyzw: QuaternionXYZW
    width_m: float
    height_m: float


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    """Latest fused telemetry state at a single simulator timestamp."""

    timestamp_ns: int
    vehicle_position_m: Point3D | None = None
    vehicle_orientation_xyzw: QuaternionXYZW | None = None
    velocity_mps: Point3D | None = None
    body_rates_rad_s: Point3D | None = None
    motor_outputs: tuple[float, float, float, float] | None = None
    active_gate_index: int | None = None
    gate_map: Mapping[int, GatePoseWorld] = field(default_factory=dict)
    sim_boot_time_ms: int | None = None
    race_start_boot_time_ms: int | None = None
    race_finish_time_ns: int | None = None
    last_gate_race_time: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate_map", MappingProxyType(dict(self.gate_map)))


@dataclass(frozen=True, slots=True)
class VisionFrame:
    """Decoded RGB frame emitted by the simulator camera stream."""

    frame_id: int
    timestamp_ns: int
    rgb: np.ndarray


@dataclass(frozen=True, slots=True)
class ProjectedGate:
    """Image-space projection of the active gate from telemetry alone."""

    gate_id: int | None
    image_polygon: tuple[Point2D, ...]
    bbox_xyxy: BBoxFloatXYXY | None
    projected_center: Point2D | None
    roi_xyxy: BBoxXYXY | None
    in_frame_fraction: float
    is_visible: bool
    is_behind_camera: bool
    is_degenerate: bool
    projection_confidence: float
    depth_range_m: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class SynchronizedSample:
    """Frame paired with the nearest usable telemetry snapshot."""

    frame: VisionFrame
    telemetry: TelemetrySnapshot
    telemetry_age_ns: int


@dataclass(frozen=True, slots=True)
class SyncDecision:
    """Synchronization outcome for a frame."""

    sample: SynchronizedSample | None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PerceptionResult:
    """Structured perception output for downstream planning/debugging."""

    frame_id: int
    timestamp_ns: int
    telemetry_timestamp_ns: int | None
    active_gate_index: int | None
    inference_path: str | None
    roi_xyxy: BBoxXYXY | None
    detection_confidence: float
    projection_confidence: float | None = None
    mask_summary: Mapping[str, float] = field(default_factory=dict)
    bbox_xyxy: BBoxXYXY | None = None
    ordered_corners: tuple[Point2D, ...] | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mask_summary", MappingProxyType(dict(self.mask_summary)))
