"""Replay raw recorded sessions through projection and detection with overlay diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2

from perception.calibration import load_camera_calibration
from perception.detector import ActiveGateDetector
from perception.projection import ProjectionEngine
from perception.storage import append_jsonl, read_json, read_jsonl, read_rgb_image, telemetry_from_dict, write_json
from perception.visualization import draw_bbox, draw_polygon, draw_text


def evaluate_replay(
    *,
    raw_session_root: str | Path,
    calibration_path: str | Path,
    weights_path: str | Path,
    output_dir: str | Path,
    device: str,
) -> Path:
    raw_root = Path(raw_session_root)
    output_dir = Path(output_dir)
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    projection_engine = ProjectionEngine(load_camera_calibration(calibration_path))
    detector = ActiveGateDetector(weights_path=weights_path, device=device)
    counters = Counter()
    results_manifest = output_dir / "results.jsonl"

    for row in read_jsonl(raw_root / "manifest.jsonl"):
        metadata = read_json(row["metadata_path"])
        telemetry = telemetry_from_dict(metadata["telemetry"])
        frame_rgb = read_rgb_image(row["image_path"])
        projected_gate = projection_engine.project_active_gate(telemetry)
        detection = detector.detect(frame_rgb, projected_gate)
        counters["frames"] += 1
        counters[f"path:{detection.inference_path}"] += 1
        if detection.failure_reason:
            counters[f"failure:{detection.failure_reason}"] += 1
        else:
            counters["detections"] += 1

        overlay = frame_rgb.copy()
        if projected_gate.image_polygon:
            overlay = draw_polygon(overlay, projected_gate.image_polygon, color_bgr=(255, 255, 0))
        overlay = draw_bbox(overlay, detection.roi_xyxy, color_bgr=(0, 255, 255))
        overlay = draw_bbox(overlay, detection.bbox_xyxy, color_bgr=(0, 255, 0))
        if detection.ordered_corners:
            overlay = draw_polygon(overlay, detection.ordered_corners, color_bgr=(0, 0, 255))
        label = detection.failure_reason or f"{detection.confidence:.3f}"
        overlay = draw_text(
            overlay,
            f"{metadata['sample_id']} | {detection.inference_path} | {label}",
        )
        cv2.imwrite(
            str(overlays_dir / f"{metadata['sample_id']}.jpg"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )
        append_jsonl(
            results_manifest,
            {
                "sample_id": metadata["sample_id"],
                "inference_path": detection.inference_path,
                "confidence": detection.confidence,
                "failure_reason": detection.failure_reason,
                "bbox_xyxy": list(detection.bbox_xyxy) if detection.bbox_xyxy else None,
            },
        )

    write_json(output_dir / "metrics.json", dict(counters))
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-session-root", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    evaluate_replay(
        raw_session_root=args.raw_session_root,
        calibration_path=args.calibration,
        weights_path=args.weights,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
