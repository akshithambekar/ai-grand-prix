from __future__ import annotations

import math
import unittest

from perception.calibration import CameraCalibration, CameraExtrinsics, CameraIntrinsics
from perception.geometry import gate_corners_world
from perception.projection import ProjectionEngine, build_padded_roi
from perception.types import GatePoseWorld, TelemetrySnapshot


class ProjectionTests(unittest.TestCase):
    def test_gate_corners_world_identity_orientation(self) -> None:
        gate = GatePoseWorld(
            gate_id=1,
            center_position_m=(0.0, 0.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            width_m=4.0,
            height_m=2.0,
        )

        corners = gate_corners_world(gate)

        expected = [
            (0.0, -2.0, -1.0),
            (0.0, 2.0, -1.0),
            (0.0, 2.0, 1.0),
            (0.0, -2.0, 1.0),
        ]
        self.assertEqual([tuple(point) for point in corners.tolist()], expected)

    def test_projection_engine_projects_gate_and_builds_roi(self) -> None:
        calibration = CameraCalibration(
            intrinsics=CameraIntrinsics(
                width_px=100,
                height_px=100,
                fx=100.0,
                fy=100.0,
                cx=50.0,
                cy=50.0,
            ),
            extrinsics=CameraExtrinsics(
                rotation_body_to_camera_xyzw=(0.0, 0.0, 0.0, 1.0),
                translation_body_to_camera_m=(0.0, 0.0, 0.0),
            ),
            sample_count=1,
            mean_reprojection_error_px=0.0,
            max_reprojection_error_px=0.0,
            rotation_std_deg=0.0,
            translation_std_m=0.0,
        )
        telemetry = TelemetrySnapshot(
            timestamp_ns=1,
            vehicle_position_m=(0.0, 0.0, 0.0),
            vehicle_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            active_gate_index=0,
            gate_map={
                0: GatePoseWorld(
                    gate_id=0,
                    center_position_m=(0.0, 0.0, 5.0),
                    orientation_xyzw=(0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)),
                    width_m=4.0,
                    height_m=2.0,
                )
            },
        )

        projected = ProjectionEngine(calibration).project_active_gate(telemetry)

        self.assertTrue(projected.is_visible)
        self.assertEqual(projected.roi_xyxy, (5, 0, 94, 100))
        self.assertAlmostEqual(projected.bbox_xyxy[0], 30.0)
        self.assertAlmostEqual(projected.bbox_xyxy[2], 70.0)
        self.assertGreater(projected.projection_confidence, 0.0)

    def test_build_padded_roi_clamps_to_image_bounds(self) -> None:
        roi = build_padded_roi(
            (-5.0, 10.0, 20.0, 30.0),
            image_width_px=40,
            image_height_px=35,
            padding_scale=0.5,
            min_padding_px=8,
        )

        self.assertEqual(roi, (0, 0, 32, 35))


if __name__ == "__main__":
    unittest.main()
