"""Mask cleanup and polygon extraction for active-gate detections."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from perception.geometry import order_points_clockwise
from perception.types import BBoxXYXY, Point2D


@dataclass(frozen=True, slots=True)
class ProcessedDetection:
    confidence: float
    roi_bbox_xyxy: BBoxXYXY
    full_frame_bbox_xyxy: BBoxXYXY
    ordered_corners: tuple[Point2D, ...]
    contour_polygon: tuple[Point2D, ...]
    mask_summary: dict[str, float]


def clean_mask(mask: np.ndarray, *, kernel_size: int = 3) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def extract_detection_from_mask(
    mask: np.ndarray,
    *,
    confidence: float,
    roi_offset_xy: tuple[int, int] = (0, 0),
) -> ProcessedDetection | None:
    cleaned = clean_mask(mask)
    contours, _hierarchy = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    if contour_area <= 0.0:
        return None

    x, y, width, height = cv2.boundingRect(contour)
    roi_bbox = (int(x), int(y), int(x + width), int(y + height))
    offset_x, offset_y = roi_offset_xy
    full_bbox = (
        roi_bbox[0] + offset_x,
        roi_bbox[1] + offset_y,
        roi_bbox[2] + offset_x,
        roi_bbox[3] + offset_y,
    )

    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.0, perimeter * 0.02)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) < 4:
        rect = cv2.minAreaRect(contour)
        approx = cv2.boxPoints(rect)

    ordered_corners = tuple(
        (point[0] + offset_x, point[1] + offset_y)
        for point in order_points_clockwise(np.asarray(approx, dtype=np.float64))
    )
    contour_points = contour.reshape(-1, 2)
    contour_polygon = tuple(
        (float(point[0] + offset_x), float(point[1] + offset_y)) for point in contour_points
    )
    moments = cv2.moments(contour)
    if moments["m00"] == 0.0:
        centroid_x = float(x + width / 2.0 + offset_x)
        centroid_y = float(y + height / 2.0 + offset_y)
    else:
        centroid_x = float((moments["m10"] / moments["m00"]) + offset_x)
        centroid_y = float((moments["m01"] / moments["m00"]) + offset_y)

    mask_summary = {
        "mask_area_px": contour_area,
        "centroid_x_px": centroid_x,
        "centroid_y_px": centroid_y,
        "bbox_width_px": float(width),
        "bbox_height_px": float(height),
    }
    return ProcessedDetection(
        confidence=confidence,
        roi_bbox_xyxy=roi_bbox,
        full_frame_bbox_xyxy=full_bbox,
        ordered_corners=ordered_corners,
        contour_polygon=contour_polygon,
        mask_summary=mask_summary,
    )


def bbox_iou(box_a: BBoxXYXY, box_b: BBoxXYXY) -> float:
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def polygon_to_mask(
    polygon_xy: tuple[Point2D, ...] | list[Point2D],
    *,
    width_px: int,
    height_px: int,
    offset_xy: tuple[int, int] = (0, 0),
) -> np.ndarray:
    mask = np.zeros((height_px, width_px), dtype=np.uint8)
    if not polygon_xy:
        return mask
    offset_x, offset_y = offset_xy
    polygon = np.asarray(
        [[point[0] - offset_x, point[1] - offset_y] for point in polygon_xy],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [polygon], 255)
    return mask
