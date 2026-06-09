"""YOLO-based active-gate detector with ROI-first inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
import numpy as np

from perception.postprocess import ProcessedDetection, bbox_iou, extract_detection_from_mask
from perception.types import Point2D, ProjectedGate


@dataclass(frozen=True, slots=True)
class DetectorResult:
    inference_path: str
    confidence: float
    roi_xyxy: tuple[int, int, int, int] | None
    bbox_xyxy: tuple[int, int, int, int] | None
    ordered_corners: tuple[Point2D, ...] | None
    mask_summary: dict[str, float]
    failure_reason: str | None


class ActiveGateDetector:
    """Wraps an Ultralytics segmentation checkpoint for active-gate inference."""

    def __init__(
        self,
        *,
        weights_path: str | Path | None = None,
        device: str = "cuda:0",
        model_repo_id: str = "openvision/yolo26-n-seg",
        model_filename: str = "model.pt",
        confidence_threshold: float = 0.25,
    ) -> None:
        from ultralytics import YOLO

        resolved_weights = (
            Path(weights_path)
            if weights_path is not None
            else Path(hf_hub_download(repo_id=model_repo_id, filename=model_filename))
        )
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._model = YOLO(str(resolved_weights))

    def detect(self, frame_rgb: np.ndarray, projection: ProjectedGate | None) -> DetectorResult:
        use_roi = projection is not None and projection.roi_xyxy is not None and projection.projection_confidence > 0.0
        if use_roi:
            roi = projection.roi_xyxy
            assert roi is not None
            crop = frame_rgb[roi[1] : roi[3], roi[0] : roi[2]]
            return self._run_inference(
                crop,
                roi_offset=(roi[0], roi[1]),
                roi_xyxy=roi,
                inference_path="roi",
                projected_gate=projection,
            )
        return self._run_inference(
            frame_rgb,
            roi_offset=(0, 0),
            roi_xyxy=None,
            inference_path="full_frame",
            projected_gate=projection,
        )

    def _run_inference(
        self,
        image_rgb: np.ndarray,
        *,
        roi_offset: tuple[int, int],
        roi_xyxy: tuple[int, int, int, int] | None,
        inference_path: str,
        projected_gate: ProjectedGate | None,
    ) -> DetectorResult:
        predictions = self._model.predict(
            source=image_rgb,
            device=self._device,
            verbose=False,
            conf=self._confidence_threshold,
        )
        if not predictions:
            return DetectorResult(
                inference_path=inference_path,
                confidence=0.0,
                roi_xyxy=roi_xyxy,
                bbox_xyxy=None,
                ordered_corners=None,
                mask_summary={},
                failure_reason="no_model_detection",
            )

        candidate = self._choose_best_prediction(
            predictions[0],
            roi_offset=roi_offset,
            projected_gate=projected_gate,
        )
        if candidate is None:
            return DetectorResult(
                inference_path=inference_path,
                confidence=0.0,
                roi_xyxy=roi_xyxy,
                bbox_xyxy=None,
                ordered_corners=None,
                mask_summary={},
                failure_reason="low_confidence_detection",
            )
        return DetectorResult(
            inference_path=inference_path,
            confidence=candidate.confidence,
            roi_xyxy=roi_xyxy,
            bbox_xyxy=candidate.full_frame_bbox_xyxy,
            ordered_corners=candidate.ordered_corners,
            mask_summary=candidate.mask_summary,
            failure_reason=None,
        )

    def _choose_best_prediction(
        self,
        prediction: Any,
        *,
        roi_offset: tuple[int, int],
        projected_gate: ProjectedGate | None,
    ) -> ProcessedDetection | None:
        if prediction.boxes is None or prediction.masks is None:
            return None

        confidences = prediction.boxes.conf.cpu().numpy()
        boxes_xyxy = prediction.boxes.xyxy.cpu().numpy()
        masks = prediction.masks.data.cpu().numpy()

        best_score = float("-inf")
        best_detection: ProcessedDetection | None = None

        for confidence, box_xyxy, mask in zip(confidences, boxes_xyxy, masks, strict=True):
            if float(confidence) < self._confidence_threshold:
                continue
            detection = extract_detection_from_mask(mask, confidence=float(confidence), roi_offset_xy=roi_offset)
            if detection is None:
                continue
            score = float(confidence)
            if projected_gate is not None and projected_gate.bbox_xyxy is not None:
                projected_bbox = tuple(int(round(value)) for value in projected_gate.bbox_xyxy)
                score += bbox_iou(detection.full_frame_bbox_xyxy, projected_bbox)
            center_prior = _projected_center_distance_bonus(detection, projected_gate)
            score += center_prior
            if score > best_score:
                best_score = score
                best_detection = detection

        return best_detection


def _projected_center_distance_bonus(
    detection: ProcessedDetection, projected_gate: ProjectedGate | None
) -> float:
    if projected_gate is None or projected_gate.projected_center is None:
        return 0.0
    centroid_x = detection.mask_summary["centroid_x_px"]
    centroid_y = detection.mask_summary["centroid_y_px"]
    dx = centroid_x - projected_gate.projected_center[0]
    dy = centroid_y - projected_gate.projected_center[1]
    distance = float(np.hypot(dx, dy))
    normalization = max(1.0, detection.mask_summary["bbox_width_px"] + detection.mask_summary["bbox_height_px"])
    return max(0.0, 1.0 - (distance / normalization))
