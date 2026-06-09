"""Simulator-first perception runtime primitives."""

from perception.calibration import CameraCalibration, CameraExtrinsics, CameraIntrinsics
from perception.detector import ActiveGateDetector
from perception.geometry import gate_corners_world
from perception.main import PerceptionRunner
from perception.projection import ProjectionEngine, ProjectionSettings
from perception.sync import TelemetrySynchronizer
from perception.telemetry import MAVLinkTelemetryReceiver, TelemetryStateCache
from perception.types import (
    GatePoseWorld,
    PerceptionResult,
    ProjectedGate,
    SynchronizedSample,
    SyncDecision,
    TelemetrySnapshot,
    VisionFrame,
)
from perception.vision import UDPVisionReceiver

__all__ = [
    "ActiveGateDetector",
    "CameraCalibration",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "GatePoseWorld",
    "MAVLinkTelemetryReceiver",
    "PerceptionResult",
    "PerceptionRunner",
    "ProjectionEngine",
    "ProjectionSettings",
    "ProjectedGate",
    "SynchronizedSample",
    "SyncDecision",
    "TelemetrySnapshot",
    "TelemetryStateCache",
    "TelemetrySynchronizer",
    "UDPVisionReceiver",
    "VisionFrame",
    "gate_corners_world",
]
