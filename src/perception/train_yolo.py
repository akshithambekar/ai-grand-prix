"""Fine-tune the active-gate segmenter on reviewed ROI crops."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
import shutil

import cv2
from huggingface_hub import hf_hub_download
import numpy as np
from ultralytics import YOLO

from perception.postprocess import bbox_iou, extract_detection_from_mask
from perception.storage import read_json, read_jsonl, write_json


def train_reviewed_dataset(
    *,
    reviewed_root: str | Path,
    output_dir: str | Path,
    device: str,
    epochs: int,
    batch: int,
    imgsz: int,
    base_repo_id: str,
    base_filename: str,
) -> Path:
    reviewed_root = Path(reviewed_root)
    output_dir = Path(output_dir)
    run_dir = output_dir / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    dataset_dir = run_dir / "dataset"
    train_images = dataset_dir / "train" / "images"
    train_labels = dataset_dir / "train" / "labels"
    val_images = dataset_dir / "val" / "images"
    val_labels = dataset_dir / "val" / "labels"
    for directory in (train_images, train_labels, val_images, val_labels):
        directory.mkdir(parents=True, exist_ok=True)

    accepted_by_session = _collect_reviewed_samples(reviewed_root)
    train_rows, val_rows, split_summary = _split_sessions(accepted_by_session)
    for row in train_rows:
        _link_or_copy(row["roi_image_path"], train_images / f"{row['sample_id']}.jpg")
        _link_or_copy(row["roi_label_path"], train_labels / f"{row['sample_id']}.txt")
    effective_val_rows = val_rows if val_rows else train_rows
    for row in effective_val_rows:
        _link_or_copy(row["roi_image_path"], val_images / f"{row['sample_id']}.jpg")
        _link_or_copy(row["roi_label_path"], val_labels / f"{row['sample_id']}.txt")

    data_yaml = run_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_dir}",
                "train: train/images",
                "val: val/images",
                "names:",
                "  0: active_gate",
            ]
        )
        + "\n"
    )

    base_checkpoint = Path(hf_hub_download(repo_id=base_repo_id, filename=base_filename))
    model = YOLO(str(base_checkpoint))
    train_results = model.train(
        data=str(data_yaml),
        task="segment",
        device=device,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        single_cls=True,
        project=str(run_dir / "ultralytics"),
        name="gate_seg",
        exist_ok=True,
    )
    val_results = model.val(data=str(data_yaml), split="val", device=device, plots=False)
    best_weights = Path(train_results.save_dir) / "weights" / "best.pt"

    evaluation_rows = effective_val_rows
    evaluation_metrics = evaluate_trained_model(
        weights_path=best_weights,
        sample_rows=evaluation_rows,
        device=device,
    )
    write_json(
        run_dir / "metrics.json",
        {
            "split_summary": split_summary,
            "best_weights": str(best_weights),
            "ultralytics_val_metrics": getattr(val_results, "results_dict", {}),
            "evaluation_metrics": evaluation_metrics,
        },
    )
    return best_weights


def evaluate_trained_model(
    *,
    weights_path: str | Path,
    sample_rows: list[dict[str, object]],
    device: str,
) -> dict[str, float]:
    model = YOLO(str(weights_path))
    bbox_ious: list[float] = []
    mask_ious: list[float] = []
    center_errors: list[float] = []
    misses = 0
    total = 0

    for row in sample_rows:
        total += 1
        metadata = read_json(row["metadata_path"])
        prediction = model.predict(source=str(row["roi_image_path"]), device=device, verbose=False, conf=0.25)[0]
        if prediction.masks is None or prediction.boxes is None or len(prediction.boxes) == 0:
            misses += 1
            continue

        confidence = float(prediction.boxes.conf.cpu().numpy()[0])
        mask = prediction.masks.data.cpu().numpy()[0] > 0.5
        detected = extract_detection_from_mask(mask, confidence=confidence)
        if detected is None:
            misses += 1
            continue

        ground_truth_polygon = tuple((float(point[0]), float(point[1])) for point in metadata["roi_polygon_px"])
        gt_x = [point[0] for point in ground_truth_polygon]
        gt_y = [point[1] for point in ground_truth_polygon]
        gt_bbox = (int(min(gt_x)), int(min(gt_y)), int(max(gt_x)), int(max(gt_y)))
        bbox_ious.append(bbox_iou(detected.roi_bbox_xyxy, gt_bbox))
        gt_mask = cv2.imread(str(row["roi_mask_path"]), cv2.IMREAD_GRAYSCALE)
        if gt_mask is not None:
            gt_mask_bool = gt_mask > 0
            intersection = float(np.logical_and(mask, gt_mask_bool).sum())
            union = float(np.logical_or(mask, gt_mask_bool).sum())
            mask_ious.append(intersection / union if union > 0.0 else 0.0)
        center_errors.append(
            float(
                np.hypot(
                    detected.mask_summary["centroid_x_px"] - np.mean(gt_x),
                    detected.mask_summary["centroid_y_px"] - np.mean(gt_y),
                )
            )
        )

    return {
        "mask_iou_mean": float(np.mean(mask_ious)) if mask_ious else 0.0,
        "bbox_iou_mean": float(np.mean(bbox_ious)) if bbox_ious else 0.0,
        "center_error_px_mean": float(np.mean(center_errors)) if center_errors else 0.0,
        "miss_rate": float(misses / total) if total else 0.0,
    }


def _collect_reviewed_samples(reviewed_root: Path) -> dict[str, list[dict[str, object]]]:
    samples_by_session: dict[str, list[dict[str, object]]] = defaultdict(list)
    manifests = reviewed_root.glob("*/manifest.jsonl") if (reviewed_root / "manifest.jsonl").exists() is False else [reviewed_root / "manifest.jsonl"]
    for manifest_path in manifests:
        for row in read_jsonl(manifest_path):
            if row["status"] != "accepted":
                continue
            samples_by_session[str(row["session_id"])].append(row)
    return samples_by_session


def _split_sessions(
    samples_by_session: dict[str, list[dict[str, object]]], val_fraction: float = 0.2
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    session_ids = sorted(samples_by_session)
    if len(session_ids) <= 1:
        only_rows = [row for rows in samples_by_session.values() for row in rows]
        return only_rows, [], {"train_sessions": session_ids, "val_sessions": [], "leakage_warning": True}

    val_count = max(1, int(round(len(session_ids) * val_fraction)))
    val_sessions = set(session_ids[-val_count:])
    train_rows = [row for session_id, rows in samples_by_session.items() if session_id not in val_sessions for row in rows]
    val_rows = [row for session_id, rows in samples_by_session.items() if session_id in val_sessions for row in rows]
    return train_rows, val_rows, {"train_sessions": [sid for sid in session_ids if sid not in val_sessions], "val_sessions": sorted(val_sessions), "leakage_warning": False}


def _link_or_copy(source: str | Path, target: str | Path) -> None:
    source_path = Path(source)
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return
    try:
        target_path.hardlink_to(source_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-root", required=True)
    parser.add_argument("--output-dir", default="runs/perception")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--base-repo-id", default="openvision/yolo26-n-seg")
    parser.add_argument("--base-filename", default="model.pt")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    train_reviewed_dataset(
        reviewed_root=args.reviewed_root,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        base_repo_id=args.base_repo_id,
        base_filename=args.base_filename,
    )


if __name__ == "__main__":
    main()
