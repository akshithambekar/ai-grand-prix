from __future__ import annotations

import unittest

import numpy as np

from perception.sync import TelemetrySynchronizer
from perception.telemetry import TelemetryStateCache
from perception.types import VisionFrame


class TelemetrySynchronizerTests(unittest.TestCase):
    def test_reports_missing_telemetry_when_cache_is_empty(self) -> None:
        cache = TelemetryStateCache()
        synchronizer = TelemetrySynchronizer(cache, max_telemetry_age_ns=5_000_000)
        frame = VisionFrame(frame_id=1, timestamp_ns=10_000_000, rgb=np.zeros((2, 2, 3), dtype=np.uint8))

        decision = synchronizer.synchronize(frame)

        self.assertIsNone(decision.sample)
        self.assertEqual(decision.failure_reason, "missing_telemetry")

    def test_uses_nearest_prior_snapshot_within_age_budget(self) -> None:
        cache = TelemetryStateCache()
        cache.update_pose(timestamp_ns=10_000_000, position_m=(1.0, 2.0, 3.0))
        cache.update_pose(timestamp_ns=15_000_000, position_m=(4.0, 5.0, 6.0))
        synchronizer = TelemetrySynchronizer(cache, max_telemetry_age_ns=6_000_000)
        frame = VisionFrame(frame_id=3, timestamp_ns=18_000_000, rgb=np.zeros((2, 2, 3), dtype=np.uint8))

        decision = synchronizer.synchronize(frame)

        self.assertIsNotNone(decision.sample)
        assert decision.sample is not None
        self.assertEqual(decision.sample.telemetry.timestamp_ns, 15_000_000)
        self.assertEqual(decision.sample.telemetry_age_ns, 3_000_000)

    def test_rejects_stale_telemetry(self) -> None:
        cache = TelemetryStateCache()
        cache.update_pose(timestamp_ns=10_000_000, position_m=(1.0, 2.0, 3.0))
        synchronizer = TelemetrySynchronizer(cache, max_telemetry_age_ns=1_000_000)
        frame = VisionFrame(frame_id=4, timestamp_ns=20_000_000, rgb=np.zeros((2, 2, 3), dtype=np.uint8))

        decision = synchronizer.synchronize(frame)

        self.assertIsNone(decision.sample)
        self.assertEqual(decision.failure_reason, "stale_telemetry")


if __name__ == "__main__":
    unittest.main()
