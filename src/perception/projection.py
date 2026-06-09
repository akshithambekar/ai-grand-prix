"""Active-gate projection and ROI policy built on the calibrated simulator camera."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception.calibration import CameraCalibration
from perception.geometry import bbox_from_points, body_points_to_camera, gate_corners_world, world_points_to_body
from perception.types import GatePoseWorld, ProjectedGate, TelemetrySnapshot


@dataclass(frozen=True, slots=True)
class ProjectionSettings:
    roi_padding_scale: float = 0.25
    roi_min_padding_px: int = 24
    min_bbox_area_px: float = 64.0
    min_in_frame_fraction: float = 0.35
    max_bbox_fraction: float = 0.9


class ProjectionEngine:
    """Projects the telemetry-selected active gate into image space."""

    def __init__(
        self,
        calibration: CameraCalibration,
        *,
        settings: ProjectionSettings | None = None,
    ) -> None:
        self._calibration = calibration
        self._settings = settings or ProjectionSettings()

    @property
    def calibration(self) -> CameraCalibration:
        return self._calibration

    def project_active_gate(self, telemetry: TelemetrySnapshot) -> ProjectedGate:
        gate_pose = _resolve_active_gate_pose(telemetry)
        if gate_pose is None:
            return ProjectedGate(
                gate_id=telemetry.active_gate_index,
                image_polygon=(),
                bbox_xyxy=None,
                projected_center=None,
                roi_xyxy=None,
                in_frame_fraction=0.0,
                is_visible=False,
                is_behind_camera=False,
                is_degenerate=True,
                projection_confidence=0.0,
            )
        return self.project_gate(telemetry, gate_pose)

    def project_gate(self, telemetry: TelemetrySnapshot, gate_pose: GatePoseWorld) -> ProjectedGate:
        if telemetry.vehicle_position_m is None or telemetry.vehicle_orientation_xyzw is None:
            return ProjectedGate(
                gate_id=gate_pose.gate_id,
                image_polygon=(),
                bbox_xyxy=None,
                projected_center=None,
                roi_xyxy=None,
                in_frame_fraction=0.0,
                is_visible=False,
                is_behind_camera=False,
                is_degenerate=True,
                projection_confidence=0.0,
            )

        world_corners = gate_corners_world(gate_pose)
        body_corners = world_points_to_body(
            world_corners,
            vehicle_position_m=telemetry.vehicle_position_m,
            vehicle_orientation_xyzw=telemetry.vehicle_orientation_xyzw,
        )
        camera_corners = body_points_to_camera(
            body_corners,
            rotation_body_to_camera_xyzw=self._calibration.extrinsics.rotation_body_to_camera_xyzw,
            translation_body_to_camera_m=self._calibration.extrinsics.translation_body_to_camera_m,
        )
        depths = camera_corners[:, 2]
        is_behind_camera = bool(np.any(depths <= 1e-4))
        image_polygon = tuple(self._project_camera_points(camera_corners))
        if len(image_polygon) < 4:
            return ProjectedGate(
                gate_id=gate_pose.gate_id,
                image_polygon=image_polygon,
                bbox_xyxy=None,
                projected_center=None,
                roi_xyxy=None,
                in_frame_fraction=0.0,
                is_visible=False,
                is_behind_camera=is_behind_camera,
                is_degenerate=True,
                projection_confidence=0.0,
                depth_range_m=(float(np.min(depths)), float(np.max(depths))),
            )

        polygon_array = np.asarray(image_polygon, dtype=np.float64)
        bbox = bbox_from_points(polygon_array)
        bbox_width = max(0.0, bbox[2] - bbox[0])
        bbox_height = max(0.0, bbox[3] - bbox[1])
        bbox_area = bbox_width * bbox_height
        image_area = self._calibration.intrinsics.width_px * self._calibration.intrinsics.height_px
        in_frame_fraction = _estimate_in_frame_fraction(
            polygon_array,
            width_px=self._calibration.intrinsics.width_px,
            height_px=self._calibration.intrinsics.height_px,
        )
        is_degenerate = bbox_area < self._settings.min_bbox_area_px
        oversized = bbox_area > image_area * self._settings.max_bbox_fraction
        projected_center = (
            float(np.mean(polygon_array[:, 0])),
            float(np.mean(polygon_array[:, 1])),
        )
        roi_xyxy = build_padded_roi(
            bbox,
            image_width_px=self._calibration.intrinsics.width_px,
            image_height_px=self._calibration.intrinsics.height_px,
            padding_scale=self._settings.roi_padding_scale,
            min_padding_px=self._settings.roi_min_padding_px,
        )
        confidence = _projection_confidence(
            behind_camera=is_behind_camera,
            is_degenerate=is_degenerate or oversized,
            in_frame_fraction=in_frame_fraction,
            bbox_fraction=bbox_area / image_area if image_area else 1.0,
            min_in_frame_fraction=self._settings.min_in_frame_fraction,
            max_bbox_fraction=self._settings.max_bbox_fraction,
        )
        return ProjectedGate(
            gate_id=gate_pose.gate_id,
            image_polygon=image_polygon,
            bbox_xyxy=bbox,
            projected_center=projected_center,
            roi_xyxy=roi_xyxy if confidence > 0.0 else None,
            in_frame_fraction=in_frame_fraction,
            is_visible=confidence > 0.0,
            is_behind_camera=is_behind_camera,
            is_degenerate=is_degenerate or oversized,
            projection_confidence=confidence,
            depth_range_m=(float(np.min(depths)), float(np.max(depths))),
        )

    def _project_camera_points(self, camera_points: np.ndarray) -> list[tuple[float, float]]:
        z = camera_points[:, 2]
        x = camera_points[:, 0]
        y = camera_points[:, 1]
        fx = self._calibration.intrinsics.fx
        fy = self._calibration.intrinsics.fy
        cx = self._calibration.intrinsics.cx
        cy = self._calibration.intrinsics.cy
        return [
            (float((fx * x_i / z_i) + cx), float((fy * y_i / z_i) + cy))
            for x_i, y_i, z_i in zip(x, y, z, strict=True)
            if z_i > 1e-4
        ]


def build_padded_roi(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    image_width_px: int,
    image_height_px: int,
    padding_scale: float,
    min_padding_px: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    width = x2 - x1
    height = y2 - y1
    padding = max(min_padding_px, int(round(max(width, height) * padding_scale)))
    left = max(0, int(np.floor(x1 - padding)))
    top = max(0, int(np.floor(y1 - padding)))
    right = min(image_width_px, int(np.ceil(x2 + padding)))
    bottom = min(image_height_px, int(np.ceil(y2 + padding)))
    return (left, top, right, bottom)


def _resolve_active_gate_pose(telemetry: TelemetrySnapshot) -> GatePoseWorld | None:
    if telemetry.active_gate_index is None:
        return None
    if telemetry.active_gate_index in telemetry.gate_map:
        return telemetry.gate_map[telemetry.active_gate_index]
    ordered_gate_ids = sorted(telemetry.gate_map)
    if 0 <= telemetry.active_gate_index < len(ordered_gate_ids):
        return telemetry.gate_map[ordered_gate_ids[telemetry.active_gate_index]]
    return None


def _estimate_in_frame_fraction(
    polygon_xy: np.ndarray, *, width_px: int, height_px: int
) -> float:
    clamped = polygon_xy.copy()
    clamped[:, 0] = np.clip(clamped[:, 0], 0.0, float(width_px))
    clamped[:, 1] = np.clip(clamped[:, 1], 0.0, float(height_px))
    total_width = max(float(np.max(polygon_xy[:, 0]) - np.min(polygon_xy[:, 0])), 1e-6)
    total_height = max(float(np.max(polygon_xy[:, 1]) - np.min(polygon_xy[:, 1])), 1e-6)
    clamped_width = float(np.max(clamped[:, 0]) - np.min(clamped[:, 0]))
    clamped_height = float(np.max(clamped[:, 1]) - np.min(clamped[:, 1]))
    return max(0.0, min(1.0, (clamped_width * clamped_height) / (total_width * total_height)))


def _projection_confidence(
    *,
    behind_camera: bool,
    is_degenerate: bool,
    in_frame_fraction: float,
    bbox_fraction: float,
    min_in_frame_fraction: float,
    max_bbox_fraction: float,
) -> float:
    if behind_camera or is_degenerate or in_frame_fraction < min_in_frame_fraction:
        return 0.0
    oversize_penalty = max(0.0, min(1.0, 1.0 - max(0.0, bbox_fraction - max_bbox_fraction)))
    return float(max(0.0, min(1.0, in_frame_fraction * oversize_penalty)))
