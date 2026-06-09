"""Camera calibration config and fitting utilities for simulator perception."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from perception.geometry import (
    body_points_to_camera,
    gate_corners_world,
    normalize_quaternion,
    quaternion_xyzw_to_rotation_matrix,
    rotation_matrix_to_quaternion_xyzw,
    world_points_to_body,
)
from perception.storage import read_json, read_jsonl, write_json
from perception.types import GatePoseWorld, QuaternionXYZW


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    width_px: int
    height_px: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_coeffs: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class CameraExtrinsics:
    rotation_body_to_camera_xyzw: QuaternionXYZW
    translation_body_to_camera_m: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    sample_count: int
    mean_reprojection_error_px: float
    max_reprojection_error_px: float
    rotation_std_deg: float
    translation_std_m: float


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    sample_id: str
    image_size_px: tuple[int, int]
    vehicle_position_m: tuple[float, float, float]
    vehicle_orientation_xyzw: QuaternionXYZW
    gate_pose: GatePoseWorld
    image_corners_px: tuple[tuple[float, float], ...]


def save_camera_calibration(calibration: CameraCalibration, output_path: str | Path) -> None:
    write_json(output_path, calibration_to_dict(calibration))


def load_camera_calibration(path: str | Path) -> CameraCalibration:
    return calibration_from_dict(read_json(path))


def calibration_to_dict(calibration: CameraCalibration) -> dict[str, Any]:
    return asdict(calibration)


def calibration_from_dict(data: dict[str, Any]) -> CameraCalibration:
    intrinsics = CameraIntrinsics(**data["intrinsics"])
    extrinsics = CameraExtrinsics(**data["extrinsics"])
    return CameraCalibration(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        sample_count=int(data["sample_count"]),
        mean_reprojection_error_px=float(data["mean_reprojection_error_px"]),
        max_reprojection_error_px=float(data["max_reprojection_error_px"]),
        rotation_std_deg=float(data.get("rotation_std_deg", 0.0)),
        translation_std_m=float(data.get("translation_std_m", 0.0)),
    )


def fit_camera_calibration(samples: list[CalibrationSample]) -> CameraCalibration:
    if not samples:
        msg = "at least one calibration sample is required"
        raise ValueError(msg)

    first_width, first_height = samples[0].image_size_px
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []

    for sample in samples:
        if sample.image_size_px != (first_width, first_height):
            msg = "all calibration samples must share the same image size"
            raise ValueError(msg)
        world_corners = gate_corners_world(sample.gate_pose)
        body_corners = world_points_to_body(
            world_corners,
            vehicle_position_m=sample.vehicle_position_m,
            vehicle_orientation_xyzw=sample.vehicle_orientation_xyzw,
        )
        object_points.append(body_corners.astype(np.float32))
        image_points.append(np.asarray(sample.image_corners_px, dtype=np.float32))

    focal_guess = float(max(first_width, first_height))
    camera_matrix = np.array(
        [
            [focal_guess, 0.0, first_width * 0.5],
            [0.0, focal_guess, first_height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros((5, 1), dtype=np.float64)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST
    flags |= cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
    rms_error, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        (first_width, first_height),
        camera_matrix,
        distortion,
        flags=flags,
    )

    rotations = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]
    mean_rotation = _average_rotation_matrix(rotations)
    extrinsic_quaternion = rotation_matrix_to_quaternion_xyzw(mean_rotation)
    translations = np.asarray([tvec.reshape(3) for tvec in tvecs], dtype=np.float64)
    mean_translation = np.mean(translations, axis=0)

    reprojection_errors = []
    for sample, object_pts in zip(samples, object_points, strict=True):
        projected = project_points_body_frame(
            object_pts.astype(np.float64),
            intrinsics=CameraIntrinsics(
                width_px=first_width,
                height_px=first_height,
                fx=float(camera_matrix[0, 0]),
                fy=float(camera_matrix[1, 1]),
                cx=float(camera_matrix[0, 2]),
                cy=float(camera_matrix[1, 2]),
                distortion_coeffs=tuple(float(value) for value in distortion.reshape(-1)),
            ),
            extrinsics=CameraExtrinsics(
                rotation_body_to_camera_xyzw=extrinsic_quaternion,
                translation_body_to_camera_m=tuple(float(value) for value in mean_translation),
            ),
        )
        observed = np.asarray(sample.image_corners_px, dtype=np.float64)
        reprojection_errors.extend(np.linalg.norm(projected - observed, axis=1).tolist())

    mean_rotation_error_deg = float(
        np.mean([_rotation_delta_degrees(rotation, mean_rotation) for rotation in rotations])
    )
    translation_std_m = float(np.mean(np.linalg.norm(translations - mean_translation, axis=1)))

    return CameraCalibration(
        intrinsics=CameraIntrinsics(
            width_px=first_width,
            height_px=first_height,
            fx=float(camera_matrix[0, 0]),
            fy=float(camera_matrix[1, 1]),
            cx=float(camera_matrix[0, 2]),
            cy=float(camera_matrix[1, 2]),
            distortion_coeffs=tuple(float(value) for value in distortion.reshape(-1)),
        ),
        extrinsics=CameraExtrinsics(
            rotation_body_to_camera_xyzw=extrinsic_quaternion,
            translation_body_to_camera_m=tuple(float(value) for value in mean_translation),
        ),
        sample_count=len(samples),
        mean_reprojection_error_px=float(np.mean(reprojection_errors)) if reprojection_errors else rms_error,
        max_reprojection_error_px=float(np.max(reprojection_errors)) if reprojection_errors else rms_error,
        rotation_std_deg=mean_rotation_error_deg,
        translation_std_m=translation_std_m,
    )


def load_calibration_samples(manifest_path: str | Path) -> list[CalibrationSample]:
    samples: list[CalibrationSample] = []
    for row in read_jsonl(manifest_path):
        samples.append(
            CalibrationSample(
                sample_id=row["sample_id"],
                image_size_px=(int(row["image_size_px"][0]), int(row["image_size_px"][1])),
                vehicle_position_m=tuple(float(value) for value in row["vehicle_position_m"]),
                vehicle_orientation_xyzw=normalize_quaternion(
                    tuple(float(value) for value in row["vehicle_orientation_xyzw"])
                ),
                gate_pose=GatePoseWorld(
                    gate_id=int(row["gate_pose"]["gate_id"]),
                    center_position_m=tuple(float(value) for value in row["gate_pose"]["center_position_m"]),
                    orientation_xyzw=normalize_quaternion(
                        tuple(float(value) for value in row["gate_pose"]["orientation_xyzw"])
                    ),
                    width_m=float(row["gate_pose"]["width_m"]),
                    height_m=float(row["gate_pose"]["height_m"]),
                ),
                image_corners_px=tuple(
                    (float(point[0]), float(point[1])) for point in row["image_corners_px"]
                ),
            )
        )
    return samples


def project_points_body_frame(
    body_points: np.ndarray,
    *,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> np.ndarray:
    camera_points = body_points_to_camera(
        body_points,
        rotation_body_to_camera_xyzw=extrinsics.rotation_body_to_camera_xyzw,
        translation_body_to_camera_m=extrinsics.translation_body_to_camera_m,
    )
    rvec = cv2.Rodrigues(quaternion_xyzw_to_rotation_matrix((0.0, 0.0, 0.0, 1.0)))[0]
    tvec = np.zeros((3, 1), dtype=np.float64)
    camera_matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    image_points, _ = cv2.projectPoints(
        camera_points.astype(np.float64),
        rvec,
        tvec,
        camera_matrix,
        np.asarray(intrinsics.distortion_coeffs, dtype=np.float64),
    )
    return image_points.reshape(-1, 2)


def _average_rotation_matrix(rotations: list[np.ndarray]) -> np.ndarray:
    stacked = np.sum(rotations, axis=0)
    u, _singular, vt = np.linalg.svd(stacked)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def _rotation_delta_degrees(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    delta = rotation_a @ rotation_b.T
    trace = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace)))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSONL calibration manifest with observed gate corners.")
    parser.add_argument("--output", required=True, help="Output path for the camera calibration JSON.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    samples = load_calibration_samples(args.manifest)
    calibration = fit_camera_calibration(samples)
    save_camera_calibration(calibration, args.output)
    print(json.dumps(calibration_to_dict(calibration), indent=2))


if __name__ == "__main__":
    main()
