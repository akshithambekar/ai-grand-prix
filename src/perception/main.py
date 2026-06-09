"""Top-level perception runner wiring simulator vision and telemetry together."""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Any

from perception.calibration import load_camera_calibration
from perception.detector import ActiveGateDetector
from perception.projection import ProjectionEngine
from perception.sync import TelemetrySynchronizer
from perception.telemetry import MAVLinkTelemetryReceiver, TelemetryStateCache
from perception.types import PerceptionResult, ProjectedGate, SyncDecision, SynchronizedSample
from perception.vision import UDPVisionReceiver

_LOGGER = logging.getLogger(__name__)


class PerceptionRunner:
    """Owns simulator ingest threads and exposes the latest perception result."""

    def __init__(
        self,
        *,
        mavlink_connection: Any,
        vision_host: str = "0.0.0.0",
        vision_port: int = 5600,
        max_telemetry_age_ns: int = 100_000_000,
        calibration_path: str | Path | None = None,
        detector_weights_path: str | Path | None = None,
        detector_device: str = "cuda:0",
    ) -> None:
        self._cache = TelemetryStateCache()
        self._telemetry_receiver = MAVLinkTelemetryReceiver(mavlink_connection, self._cache)
        self._vision_receiver = UDPVisionReceiver(host=vision_host, port=vision_port)
        self._synchronizer = TelemetrySynchronizer(
            self._cache, max_telemetry_age_ns=max_telemetry_age_ns
        )
        self._projection_engine = (
            ProjectionEngine(load_camera_calibration(calibration_path))
            if calibration_path is not None
            else None
        )
        self._detector = (
            ActiveGateDetector(weights_path=detector_weights_path, device=detector_device)
            if detector_weights_path is not None
            else None
        )

        self._latest_result: PerceptionResult | None = None
        self._latest_sample: SynchronizedSample | None = None
        self._result_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self._telemetry_receiver.start()
        self._vision_receiver.start()
        self._worker_thread = threading.Thread(target=self._run, name="perception-runner")
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._vision_receiver.stop()
        self._telemetry_receiver.stop()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)
        self._vision_receiver.join(timeout=1.0)
        self._telemetry_receiver.join(timeout=1.0)

    def get_latest_result(self) -> PerceptionResult | None:
        with self._result_lock:
            return self._latest_result

    def get_latest_sample(self) -> SynchronizedSample | None:
        with self._result_lock:
            return self._latest_sample

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._vision_receiver.get_frame(timeout=0.1)
            except queue.Empty:
                continue

            decision = self._synchronizer.synchronize(frame)
            if decision.sample is None:
                result = self._build_failure_result(
                    frame_id=frame.frame_id,
                    timestamp_ns=frame.timestamp_ns,
                    decision=decision,
                )
                if decision.failure_reason == "stale_telemetry":
                    _LOGGER.info(
                        "skipping frame %s due to stale telemetry at sim timestamp %s",
                        frame.frame_id,
                        frame.timestamp_ns,
                    )
                elif decision.failure_reason == "missing_telemetry":
                    _LOGGER.info(
                        "skipping frame %s because telemetry has not arrived yet",
                        frame.frame_id,
                    )
                self._publish(result=result, sample=None)
                continue

            projected_gate = (
                self._projection_engine.project_active_gate(decision.sample.telemetry)
                if self._projection_engine is not None
                else None
            )
            result = self._build_result(decision.sample, projected_gate)
            self._publish(result=result, sample=decision.sample)

    def _publish(self, *, result: PerceptionResult, sample: SynchronizedSample | None) -> None:
        with self._result_lock:
            self._latest_result = result
            self._latest_sample = sample

    def _build_result(
        self, sample: SynchronizedSample, projected_gate: ProjectedGate | None
    ) -> PerceptionResult:
        if self._detector is None:
            failure_reason = "detector_not_configured"
            if projected_gate is not None and projected_gate.projection_confidence <= 0.0:
                failure_reason = "invalid_projection"
            return PerceptionResult(
                frame_id=sample.frame.frame_id,
                timestamp_ns=sample.frame.timestamp_ns,
                telemetry_timestamp_ns=sample.telemetry.timestamp_ns,
                active_gate_index=sample.telemetry.active_gate_index,
                inference_path=None,
                roi_xyxy=projected_gate.roi_xyxy if projected_gate is not None else None,
                detection_confidence=0.0,
                projection_confidence=projected_gate.projection_confidence if projected_gate is not None else None,
                failure_reason=failure_reason,
            )

        detection = self._detector.detect(sample.frame.rgb, projected_gate)
        return PerceptionResult(
            frame_id=sample.frame.frame_id,
            timestamp_ns=sample.frame.timestamp_ns,
            telemetry_timestamp_ns=sample.telemetry.timestamp_ns,
            active_gate_index=sample.telemetry.active_gate_index,
            inference_path=detection.inference_path,
            roi_xyxy=detection.roi_xyxy,
            detection_confidence=detection.confidence,
            projection_confidence=projected_gate.projection_confidence if projected_gate is not None else None,
            mask_summary=detection.mask_summary,
            bbox_xyxy=detection.bbox_xyxy,
            ordered_corners=detection.ordered_corners,
            failure_reason=detection.failure_reason,
        )

    def _build_failure_result(
        self, *, frame_id: int, timestamp_ns: int, decision: SyncDecision
    ) -> PerceptionResult:
        latest_snapshot = self._cache.get_latest_snapshot()
        return PerceptionResult(
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            telemetry_timestamp_ns=latest_snapshot.timestamp_ns if latest_snapshot is not None else None,
            active_gate_index=latest_snapshot.active_gate_index if latest_snapshot is not None else None,
            inference_path=None,
            roi_xyxy=None,
            detection_confidence=0.0,
            failure_reason=decision.failure_reason,
        )
