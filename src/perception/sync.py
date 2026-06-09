"""Frame-to-telemetry synchronization utilities."""

from __future__ import annotations

from perception.telemetry import TelemetryStateCache
from perception.types import SyncDecision, SynchronizedSample, VisionFrame


class TelemetrySynchronizer:
    """Align vision frames to the nearest usable telemetry snapshot."""

    def __init__(self, cache: TelemetryStateCache, *, max_telemetry_age_ns: int = 100_000_000) -> None:
        self._cache = cache
        self._max_telemetry_age_ns = max_telemetry_age_ns

    def synchronize(self, frame: VisionFrame) -> SyncDecision:
        snapshot, age_ns = self._cache.get_snapshot_near(
            target_timestamp_ns=frame.timestamp_ns or None,
            max_age_ns=self._max_telemetry_age_ns,
        )
        if snapshot is None:
            if age_ns is None:
                return SyncDecision(sample=None, failure_reason="missing_telemetry")
            return SyncDecision(sample=None, failure_reason="stale_telemetry")

        return SyncDecision(
            sample=SynchronizedSample(frame=frame, telemetry=snapshot, telemetry_age_ns=age_ns or 0)
        )
