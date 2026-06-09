"""Overlay and contact-sheet helpers for perception diagnostics."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from perception.types import BBoxXYXY, Point2D


def draw_polygon(
    image_rgb: np.ndarray,
    polygon_xy: tuple[Point2D, ...] | list[Point2D],
    *,
    color_bgr: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    output = image_rgb.copy()
    if not polygon_xy:
        return output
    points = np.asarray([[int(round(x)), int(round(y))] for x, y in polygon_xy], dtype=np.int32)
    cv2.polylines(output, [points], isClosed=True, color=color_bgr, thickness=thickness)
    return output


def draw_bbox(
    image_rgb: np.ndarray,
    bbox_xyxy: BBoxXYXY | None,
    *,
    color_bgr: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    output = image_rgb.copy()
    if bbox_xyxy is None:
        return output
    cv2.rectangle(output, bbox_xyxy[:2], bbox_xyxy[2:], color_bgr, thickness)
    return output


def draw_text(
    image_rgb: np.ndarray,
    text: str,
    *,
    origin_xy: tuple[int, int] = (16, 24),
    color_bgr: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    output = image_rgb.copy()
    cv2.putText(
        output,
        text,
        origin_xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color_bgr,
        2,
        cv2.LINE_AA,
    )
    return output


def save_contact_sheet(
    image_paths: list[Path],
    *,
    output_path: str | Path,
    columns: int = 3,
    tile_size_px: tuple[int, int] = (480, 270),
) -> None:
    if not image_paths:
        return

    tile_width, tile_height = tile_size_px
    rows = int(np.ceil(len(image_paths) / columns))
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)

    for index, image_path in enumerate(image_paths):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        resized = cv2.resize(image_bgr, tile_size_px)
        row = index // columns
        col = index % columns
        top = row * tile_height
        left = col * tile_width
        sheet[top : top + tile_height, left : left + tile_width] = resized

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), sheet)
