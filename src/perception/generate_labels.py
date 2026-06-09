"""Generate weak ROI-local segmentation labels from calibrated gate projection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from perception.calibration import load_camera_calibration
from perception.postprocess import polygon_to_mask
from perception.projection import ProjectionEngine
from perception.storage import (
    append_jsonl,
    build_sample_id,
    projected_gate_to_dict,
    read_json,
    read_jsonl,
    read_rgb_image,
    telemetry_from_dict,
    write_json,
    write_rgb_image,
)


def generate_weak_labels(
    *,
    raw_session_root: str | Path,
    calibration_path: str | Path,
    output_root: str | Path,
) -> Path:
    raw_root = Path(raw_session_root)
    output_root = Path(output_root)
    manifest_rows = read_jsonl(raw_root / "manifest.jsonl")
    session_info = read_json(raw_root / "session.json")
    session_id = str(session_info["session_id"])
    session_root = output_root / session_id
    images_dir = session_root / "images"
    masks_dir = session_root / "masks"
    labels_dir = session_root / "labels"
    metadata_dir = session_root / "metadata"
    manifest_path = session_root / "manifest.jsonl"
    for directory in (images_dir, masks_dir, labels_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    projection_engine = ProjectionEngine(load_camera_calibration(calibration_path))

    for row in manifest_rows:
        metadata = read_json(row["metadata_path"])
        telemetry = telemetry_from_dict(metadata["telemetry"])
        image_rgb = read_rgb_image(row["image_path"])
        projected_gate = projection_engine.project_active_gate(telemetry)
        sample_id = metadata["sample_id"]

        if projected_gate.roi_xyxy is None or projected_gate.projection_confidence <= 0.0:
            append_jsonl(
                manifest_path,
                {
                    "sample_id": sample_id,
                    "session_id": session_id,
                    "status": "rejected",
                    "rejection_reason": "invalid_projection",
                    "source_metadata_path": row["metadata_path"],
                },
            )
            continue

        roi = projected_gate.roi_xyxy
        assert roi is not None
        full_mask = polygon_to_mask(
            projected_gate.image_polygon,
            width_px=image_rgb.shape[1],
            height_px=image_rgb.shape[0],
        )
        roi_image = image_rgb[roi[1] : roi[3], roi[0] : roi[2]]
        roi_mask = full_mask[roi[1] : roi[3], roi[0] : roi[2]]
        roi_polygon = [
            (point[0] - roi[0], point[1] - roi[1]) for point in projected_gate.image_polygon
        ]
        if roi_image.size == 0 or roi_mask.sum() == 0:
            append_jsonl(
                manifest_path,
                {
                    "sample_id": sample_id,
                    "session_id": session_id,
                    "status": "rejected",
                    "rejection_reason": "empty_roi",
                    "source_metadata_path": row["metadata_path"],
                },
            )
            continue

        roi_image_path = images_dir / f"{sample_id}.jpg"
        roi_mask_path = masks_dir / f"{sample_id}.png"
        roi_label_path = labels_dir / f"{sample_id}.txt"
        roi_metadata_path = metadata_dir / f"{sample_id}.json"
        write_rgb_image(roi_image_path, roi_image)
        write_rgb_image(roi_mask_path, np.repeat(roi_mask[:, :, None], 3, axis=2))
        roi_label_path.write_text(_yolo_polygon_line(roi_polygon, roi_image.shape[1], roi_image.shape[0]) + "\n")
        write_json(
            roi_metadata_path,
            {
                "sample_id": sample_id,
                "session_id": session_id,
                "source_metadata_path": row["metadata_path"],
                "source_image_path": row["image_path"],
                "roi_image_path": str(roi_image_path),
                "roi_mask_path": str(roi_mask_path),
                "roi_label_path": str(roi_label_path),
                "roi_xyxy": list(roi),
                "roi_polygon_px": [[float(x), float(y)] for x, y in roi_polygon],
                "projected_gate": projected_gate_to_dict(projected_gate),
                "status": "weak_valid",
            },
        )
        append_jsonl(
            manifest_path,
            {
                "sample_id": sample_id,
                "session_id": session_id,
                "status": "weak_valid",
                "roi_image_path": str(roi_image_path),
                "roi_mask_path": str(roi_mask_path),
                "roi_label_path": str(roi_label_path),
                "metadata_path": str(roi_metadata_path),
            },
        )

    return session_root


def _yolo_polygon_line(
    polygon_xy: list[tuple[float, float]], width_px: int, height_px: int
) -> str:
    normalized = []
    for x, y in polygon_xy:
        normalized.append(max(0.0, min(1.0, x / width_px)))
        normalized.append(max(0.0, min(1.0, y / height_px)))
    return "0 " + " ".join(f"{value:.6f}" for value in normalized)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-session-root", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-root", default="artifacts/perception/weak_labels")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    generate_weak_labels(
        raw_session_root=args.raw_session_root,
        calibration_path=args.calibration,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
