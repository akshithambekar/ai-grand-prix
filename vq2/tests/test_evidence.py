import math
import struct
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.evidence import (
    CYAN_PRESENT_FRACTION,
    EvidenceWriter,
    FrameRecorder,
    ManualControlConfig,
    TelemetryRecorder,
    cyan_fraction,
    json_safe,
    map_manual_keys,
    update_throttle,
)


class ManualMappingTests(unittest.TestCase):
    def setUp(self):
        self.config = ManualControlConfig(
            hover_thrust=0.6,
            thrust_step=0.1,
            roll_rate=0.2,
            pitch_rate=0.3,
            forward_speed=2.0,
            lateral_speed=1.5,
            vertical_speed=0.75,
            motor_hover=0.5,
            motor_thrust_step=0.1,
            motor_tilt_step=0.05,
        )

    def test_attitude_wasd_qe_mapping(self):
        self.assertEqual(
            map_manual_keys({"w", "a", "q"}, "attitude", self.config).values,
            (-0.2, -0.3, 0.0, 0.7),
        )
        self.assertEqual(
            map_manual_keys({"s", "d", "e"}, "attitude", self.config).values,
            (0.2, 0.3, 0.0, 0.5),
        )

    def test_velocity_uses_body_ned_signs(self):
        self.assertEqual(
            map_manual_keys({"w", "a", "q"}, "velocity", self.config).values,
            (2.0, -1.5, -0.75),
        )
        self.assertEqual(
            map_manual_keys({"s", "d", "e"}, "velocity", self.config).values,
            (-2.0, 1.5, 0.75),
        )

    def test_opposing_keys_cancel(self):
        command = map_manual_keys({"w", "s", "a", "d", "q", "e"}, "attitude", self.config)
        self.assertEqual(command.values, (0.0, 0.0, 0.0, 0.6))

    def test_motor_mixer_is_bounded_and_directional(self):
        forward = map_manual_keys({"w"}, "motor", self.config).values
        self.assertEqual(forward, (0.45, 0.45, 0.55, 0.55))
        extreme = ManualControlConfig(motor_hover=0.99, motor_thrust_step=0.2)
        self.assertEqual(map_manual_keys({"q"}, "motor", extreme).values, (1.0, 1.0, 1.0, 1.0))

    def test_manual_throttle_ramps_persists_and_clamps(self):
        self.assertAlmostEqual(update_throttle(0.0, {"q"}, 0.1, 0.2, 0.5), 0.02)
        self.assertAlmostEqual(update_throttle(0.3, set(), 0.1, 0.2, 0.5), 0.3)
        self.assertAlmostEqual(update_throttle(0.3, {"e"}, 0.1, 0.2, 0.5), 0.28)
        self.assertEqual(update_throttle(0.49, {"q"}, 0.1, 0.2, 0.5), 0.5)


class EvidenceTests(unittest.TestCase):
    def test_cyan_fraction_detects_a_cyan_guide(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.line(image, (10, 50), (90, 50), (255, 255, 0), 4)
        self.assertGreater(cyan_fraction(image), CYAN_PRESENT_FRACTION)
        self.assertEqual(cyan_fraction(np.zeros_like(image)), 0.0)

    def test_non_finite_telemetry_becomes_json_null(self):
        self.assertEqual(json_safe({"nulled": math.nan, "valid": 1.0}), {"nulled": None, "valid": 1.0})

    def test_frame_recorder_saves_jpeg_and_cyan_result(self):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        cv2.line(image, (5, 24), (58, 24), (255, 255, 0), 5)
        encoded, jpeg = cv2.imencode(".jpg", image)
        self.assertTrue(encoded)
        jpeg_bytes = jpeg.tobytes()

        with tempfile.TemporaryDirectory() as temp_dir:
            writer = EvidenceWriter(Path(temp_dir))
            recorder = FrameRecorder("127.0.0.1", 5600, writer)
            recorder._save_frame(
                7,
                {
                    "chunks": {0: jpeg_bytes},
                    "total_chunks": 1,
                    "jpeg_size": len(jpeg_bytes),
                    "sim_time_ns": 123456789,
                },
            )
            writer.close()

            self.assertEqual(recorder.frame_count, 1)
            self.assertTrue(recorder.summary()["cyan_present"])
            self.assertEqual(len(list((Path(temp_dir) / "frames").glob("*.jpg"))), 1)

    def test_track_parser_identifies_nulled_geometry(self):
        gate = struct.pack("<Hfffffffff", 3, *(0.0 for _ in range(9)))
        payload = struct.pack("<H", 1) + gate
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = EvidenceWriter(Path(temp_dir))
            recorder = TelemetryRecorder(object(), writer)
            recorder._record_track_data(12, payload)
            track = recorder.latest("TRACK_DATA")
            writer.close()

        self.assertEqual(track["gate_count"], 1)
        self.assertFalse(track["geometry_published"])
        self.assertFalse(track["gates"][0]["geometry_published"])


if __name__ == "__main__":
    unittest.main()
