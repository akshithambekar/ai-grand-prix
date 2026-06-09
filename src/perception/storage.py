"""Filesystem and serialization helpers for perception artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from perception.types import GatePoseWorld, PerceptionResult, ProjectedGate, TelemetrySnapshot


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_rgb_image(path: str | Path, rgb: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(target), bgr):
        msg = f"failed to write image to {target}"
        raise RuntimeError(msg)


def read_rgb_image(path: str | Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        msg = f"failed to read image at {path}"
        raise RuntimeError(msg)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def gate_to_dict(gate: GatePoseWorld) -> dict[str, Any]:
    return {
        "gate_id": gate.gate_id,
        "center_position_m": list(gate.center_position_m),
        "orientation_xyzw": list(gate.orientation_xyzw),
        "width_m": gate.width_m,
        "height_m": gate.height_m,
    }


def gate_from_dict(data: dict[str, Any]) -> GatePoseWorld:
    return GatePoseWorld(
        gate_id=int(data["gate_id"]),
        center_position_m=tuple(float(value) for value in data["center_position_m"]),
        orientation_xyzw=tuple(float(value) for value in data["orientation_xyzw"]),
        width_m=float(data["width_m"]),
        height_m=float(data["height_m"]),
    )


def telemetry_to_dict(snapshot: TelemetrySnapshot) -> dict[str, Any]:
    return {
        "timestamp_ns": snapshot.timestamp_ns,
        "vehicle_position_m": list(snapshot.vehicle_position_m) if snapshot.vehicle_position_m else None,
        "vehicle_orientation_xyzw": list(snapshot.vehicle_orientation_xyzw)
        if snapshot.vehicle_orientation_xyzw
        else None,
        "velocity_mps": list(snapshot.velocity_mps) if snapshot.velocity_mps else None,
        "body_rates_rad_s": list(snapshot.body_rates_rad_s) if snapshot.body_rates_rad_s else None,
        "motor_outputs": list(snapshot.motor_outputs) if snapshot.motor_outputs else None,
        "active_gate_index": snapshot.active_gate_index,
        "gate_map": {str(gate_id): gate_to_dict(gate) for gate_id, gate in snapshot.gate_map.items()},
        "sim_boot_time_ms": snapshot.sim_boot_time_ms,
        "race_start_boot_time_ms": snapshot.race_start_boot_time_ms,
        "race_finish_time_ns": snapshot.race_finish_time_ns,
        "last_gate_race_time": snapshot.last_gate_race_time,
    }


def telemetry_from_dict(data: dict[str, Any]) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        timestamp_ns=int(data["timestamp_ns"]),
        vehicle_position_m=tuple(float(value) for value in data["vehicle_position_m"])
        if data.get("vehicle_position_m") is not None
        else None,
        vehicle_orientation_xyzw=tuple(float(value) for value in data["vehicle_orientation_xyzw"])
        if data.get("vehicle_orientation_xyzw") is not None
        else None,
        velocity_mps=tuple(float(value) for value in data["velocity_mps"])
        if data.get("velocity_mps") is not None
        else None,
        body_rates_rad_s=tuple(float(value) for value in data["body_rates_rad_s"])
        if data.get("body_rates_rad_s") is not None
        else None,
        motor_outputs=tuple(float(value) for value in data["motor_outputs"])
        if data.get("motor_outputs") is not None
        else None,
        active_gate_index=data.get("active_gate_index"),
        gate_map={int(key): gate_from_dict(value) for key, value in data.get("gate_map", {}).items()},
        sim_boot_time_ms=data.get("sim_boot_time_ms"),
        race_start_boot_time_ms=data.get("race_start_boot_time_ms"),
        race_finish_time_ns=data.get("race_finish_time_ns"),
        last_gate_race_time=data.get("last_gate_race_time"),
    )


def projected_gate_to_dict(projected_gate: ProjectedGate | None) -> dict[str, Any] | None:
    if projected_gate is None:
        return None
    return {
        "gate_id": projected_gate.gate_id,
        "image_polygon": [[float(x), float(y)] for x, y in projected_gate.image_polygon],
        "bbox_xyxy": list(projected_gate.bbox_xyxy) if projected_gate.bbox_xyxy else None,
        "projected_center": list(projected_gate.projected_center)
        if projected_gate.projected_center
        else None,
        "roi_xyxy": list(projected_gate.roi_xyxy) if projected_gate.roi_xyxy else None,
        "in_frame_fraction": projected_gate.in_frame_fraction,
        "is_visible": projected_gate.is_visible,
        "is_behind_camera": projected_gate.is_behind_camera,
        "is_degenerate": projected_gate.is_degenerate,
        "projection_confidence": projected_gate.projection_confidence,
        "depth_range_m": list(projected_gate.depth_range_m) if projected_gate.depth_range_m else None,
    }


def projected_gate_from_dict(data: dict[str, Any] | None) -> ProjectedGate | None:
    if data is None:
        return None
    return ProjectedGate(
        gate_id=data.get("gate_id"),
        image_polygon=tuple((float(point[0]), float(point[1])) for point in data["image_polygon"]),
        bbox_xyxy=tuple(float(value) for value in data["bbox_xyxy"])
        if data.get("bbox_xyxy") is not None
        else None,
        projected_center=tuple(float(value) for value in data["projected_center"])
        if data.get("projected_center") is not None
        else None,
        roi_xyxy=tuple(int(value) for value in data["roi_xyxy"])
        if data.get("roi_xyxy") is not None
        else None,
        in_frame_fraction=float(data["in_frame_fraction"]),
        is_visible=bool(data["is_visible"]),
        is_behind_camera=bool(data["is_behind_camera"]),
        is_degenerate=bool(data["is_degenerate"]),
        projection_confidence=float(data["projection_confidence"]),
        depth_range_m=tuple(float(value) for value in data["depth_range_m"])
        if data.get("depth_range_m") is not None
        else None,
    )


def perception_result_to_dict(result: PerceptionResult) -> dict[str, Any]:
    return {
        "frame_id": result.frame_id,
        "timestamp_ns": result.timestamp_ns,
        "telemetry_timestamp_ns": result.telemetry_timestamp_ns,
        "active_gate_index": result.active_gate_index,
        "inference_path": result.inference_path,
        "roi_xyxy": list(result.roi_xyxy) if result.roi_xyxy else None,
        "detection_confidence": result.detection_confidence,
        "projection_confidence": result.projection_confidence,
        "mask_summary": dict(result.mask_summary),
        "bbox_xyxy": list(result.bbox_xyxy) if result.bbox_xyxy else None,
        "ordered_corners": [[float(x), float(y)] for x, y in result.ordered_corners]
        if result.ordered_corners
        else None,
        "failure_reason": result.failure_reason,
    }


def build_sample_id(frame_id: int, timestamp_ns: int) -> str:
    return f"{frame_id:08d}_{timestamp_ns}"
