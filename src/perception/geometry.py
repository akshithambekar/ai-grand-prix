"""Geometry helpers shared across calibration, projection, and postprocessing."""

from __future__ import annotations

import math

import numpy as np

from perception.types import GatePoseWorld, Point2D, QuaternionXYZW


def quaternion_xyzw_to_rotation_matrix(quaternion_xyzw: QuaternionXYZW) -> np.ndarray:
    x, y, z, w = quaternion_xyzw
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> QuaternionXYZW:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    return normalize_quaternion((x, y, z, w))


def normalize_quaternion(quaternion_xyzw: QuaternionXYZW) -> QuaternionXYZW:
    vector = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        msg = "cannot normalize zero-length quaternion"
        raise ValueError(msg)
    normalized = vector / norm
    return tuple(float(component) for component in normalized)  # type: ignore[return-value]


def gate_corners_world(gate_pose: GatePoseWorld) -> np.ndarray:
    half_width = gate_pose.width_m * 0.5
    half_height = gate_pose.height_m * 0.5
    local_corners = np.array(
        [
            [0.0, -half_width, -half_height],
            [0.0, half_width, -half_height],
            [0.0, half_width, half_height],
            [0.0, -half_width, half_height],
        ],
        dtype=np.float64,
    )
    rotation = quaternion_xyzw_to_rotation_matrix(gate_pose.orientation_xyzw)
    center = np.asarray(gate_pose.center_position_m, dtype=np.float64)
    return (rotation @ local_corners.T).T + center


def world_points_to_body(
    points_world: np.ndarray,
    *,
    vehicle_position_m: tuple[float, float, float],
    vehicle_orientation_xyzw: QuaternionXYZW,
) -> np.ndarray:
    rotation_body_to_world = quaternion_xyzw_to_rotation_matrix(vehicle_orientation_xyzw)
    translation_world = np.asarray(vehicle_position_m, dtype=np.float64)
    shifted = np.asarray(points_world, dtype=np.float64) - translation_world
    return shifted @ rotation_body_to_world


def body_points_to_camera(
    points_body: np.ndarray,
    *,
    rotation_body_to_camera_xyzw: QuaternionXYZW,
    translation_body_to_camera_m: tuple[float, float, float],
) -> np.ndarray:
    rotation = quaternion_xyzw_to_rotation_matrix(rotation_body_to_camera_xyzw)
    translation = np.asarray(translation_body_to_camera_m, dtype=np.float64)
    return (rotation @ np.asarray(points_body, dtype=np.float64).T).T + translation


def bbox_from_points(points_xy: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(np.min(points_xy[:, 0])),
        float(np.min(points_xy[:, 1])),
        float(np.max(points_xy[:, 0])),
        float(np.max(points_xy[:, 1])),
    )


def polygon_area(points_xy: np.ndarray) -> float:
    if len(points_xy) < 3:
        return 0.0
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def order_points_clockwise(points_xy: np.ndarray) -> tuple[Point2D, ...]:
    points = np.asarray(points_xy, dtype=np.float64)
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = points[order]
    start_index = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    rolled = np.roll(ordered, -start_index, axis=0)
    return tuple((float(point[0]), float(point[1])) for point in rolled)
