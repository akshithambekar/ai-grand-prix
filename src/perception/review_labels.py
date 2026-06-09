"""Render review overlays and apply accept/reject/flag decisions to weak labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import shutil

import cv2
import numpy as np

from perception.postprocess import polygon_to_mask
from perception.storage import append_jsonl, read_json, read_jsonl, write_json, write_jsonl
from perception.visualization import draw_bbox, draw_polygon, draw_text, save_contact_sheet


def render_review_assets(*, weak_session_root: str | Path, output_dir: str | Path, per_sheet: int = 12) -> Path:
    weak_root = Path(weak_session_root)
    output_dir = Path(output_dir)
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_jsonl(weak_root / "manifest.jsonl")
    overlay_paths: list[Path] = []
    decision_rows: list[dict[str, object]] = []

    for row in manifest_rows:
        if row["status"] != "weak_valid":
            decision_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "status": "reject",
                    "reason": row.get("rejection_reason", "invalid_weak_label"),
                }
            )
            continue

        metadata = read_json(row["metadata_path"])
        source_image = cv2.imread(str(metadata["source_image_path"]), cv2.IMREAD_COLOR)
        roi = tuple(int(value) for value in metadata["roi_xyxy"])
        polygon = tuple((float(point[0]), float(point[1])) for point in metadata["roi_polygon_px"])
        full_polygon = tuple((point[0] + roi[0], point[1] + roi[1]) for point in polygon)
        overlay = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        overlay = draw_polygon(overlay, full_polygon, color_bgr=(0, 255, 0), thickness=2)
        overlay = draw_bbox(overlay, roi, color_bgr=(255, 255, 0), thickness=2)
        overlay = draw_text(overlay, f"{metadata['sample_id']} | weak_valid")
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        overlay_path = overlays_dir / f"{metadata['sample_id']}.jpg"
        cv2.imwrite(str(overlay_path), overlay_bgr)
        overlay_paths.append(overlay_path)
        decision_rows.append({"sample_id": metadata["sample_id"], "status": "accept", "reason": ""})

    decisions_path = output_dir / "review_decisions_template.jsonl"
    write_jsonl(decisions_path, decision_rows)

    for index in range(0, len(overlay_paths), per_sheet):
        chunk = overlay_paths[index : index + per_sheet]
        save_contact_sheet(
            chunk,
            output_path=output_dir / "contact_sheets" / f"sheet_{index // per_sheet:03d}.jpg",
            columns=3,
        )

    return decisions_path


def apply_review_decisions(
    *,
    weak_session_root: str | Path,
    decisions_path: str | Path,
    output_root: str | Path,
) -> Path:
    weak_root = Path(weak_session_root)
    output_root = Path(output_root)
    session_id = weak_root.name
    reviewed_root = output_root / session_id
    for dirname in ("images", "masks", "labels", "metadata"):
        (reviewed_root / dirname).mkdir(parents=True, exist_ok=True)

    manifest_rows = {row["sample_id"]: row for row in read_jsonl(weak_root / "manifest.jsonl")}
    decisions = {row["sample_id"]: row for row in read_jsonl(decisions_path)}
    reviewed_manifest_path = reviewed_root / "manifest.jsonl"

    summary = defaultdict(int)
    for sample_id, decision in decisions.items():
        weak_row = manifest_rows.get(sample_id)
        if weak_row is None:
            continue
        summary[str(decision["status"])] += 1
        if weak_row["status"] != "weak_valid":
            append_jsonl(
                reviewed_manifest_path,
                {
                    "sample_id": sample_id,
                    "session_id": session_id,
                    "status": "rejected",
                    "reason": weak_row.get("rejection_reason", "invalid_weak_label"),
                },
            )
            continue

        metadata = read_json(weak_row["metadata_path"])
        status = str(decision["status"])
        if status == "reject":
            append_jsonl(
                reviewed_manifest_path,
                {
                    "sample_id": sample_id,
                    "session_id": session_id,
                    "status": "rejected",
                    "reason": decision.get("reason", "user_rejected"),
                },
            )
            continue
        if status == "flag":
            append_jsonl(
                reviewed_manifest_path,
                {
                    "sample_id": sample_id,
                    "session_id": session_id,
                    "status": "flagged",
                    "reason": decision.get("reason", "needs_correction"),
                },
            )
            continue

        corrected_polygon = decision.get("corrected_roi_polygon_px")
        if corrected_polygon:
            polygon = tuple((float(point[0]), float(point[1])) for point in corrected_polygon)
            roi = tuple(int(value) for value in metadata["roi_xyxy"])
            roi_image = cv2.imread(str(metadata["roi_image_path"]), cv2.IMREAD_COLOR)
            roi_mask = polygon_to_mask(
                polygon,
                width_px=roi_image.shape[1],
                height_px=roi_image.shape[0],
            )
            roi_mask_path = reviewed_root / "masks" / f"{sample_id}.png"
            cv2.imwrite(str(roi_mask_path), roi_mask)
            roi_label_path = reviewed_root / "labels" / f"{sample_id}.txt"
            roi_label_path.write_text(_yolo_polygon_line(polygon, roi_image.shape[1], roi_image.shape[0]) + "\n")
            reviewed_metadata = dict(metadata)
            reviewed_metadata["roi_polygon_px"] = [[float(x), float(y)] for x, y in polygon]
            reviewed_metadata["roi_mask_path"] = str(roi_mask_path)
            reviewed_metadata["roi_label_path"] = str(roi_label_path)
        else:
            roi_mask_path = reviewed_root / "masks" / f"{sample_id}.png"
            roi_label_path = reviewed_root / "labels" / f"{sample_id}.txt"
            shutil.copy2(metadata["roi_mask_path"], roi_mask_path)
            shutil.copy2(metadata["roi_label_path"], roi_label_path)
            reviewed_metadata = dict(metadata)
            reviewed_metadata["roi_mask_path"] = str(roi_mask_path)
            reviewed_metadata["roi_label_path"] = str(roi_label_path)

        reviewed_image_path = reviewed_root / "images" / f"{sample_id}.jpg"
        reviewed_metadata_path = reviewed_root / "metadata" / f"{sample_id}.json"
        shutil.copy2(metadata["roi_image_path"], reviewed_image_path)
        reviewed_metadata["roi_image_path"] = str(reviewed_image_path)
        reviewed_metadata["review_status"] = "accepted"
        write_json(reviewed_metadata_path, reviewed_metadata)
        append_jsonl(
            reviewed_manifest_path,
            {
                "sample_id": sample_id,
                "session_id": session_id,
                "status": "accepted",
                "roi_image_path": str(reviewed_image_path),
                "roi_mask_path": str(roi_mask_path),
                "roi_label_path": str(roi_label_path),
                "metadata_path": str(reviewed_metadata_path),
            },
        )

    write_json(reviewed_root / "summary.json", dict(summary))
    return reviewed_root


def _yolo_polygon_line(
    polygon_xy: tuple[tuple[float, float], ...], width_px: int, height_px: int
) -> str:
    values: list[float] = []
    for x, y in polygon_xy:
        values.append(max(0.0, min(1.0, x / width_px)))
        values.append(max(0.0, min(1.0, y / height_px)))
    return "0 " + " ".join(f"{value:.6f}" for value in values)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--weak-session-root", required=True)
    render_parser.add_argument("--output-dir", required=True)
    render_parser.add_argument("--per-sheet", type=int, default=12)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--weak-session-root", required=True)
    apply_parser.add_argument("--decisions", required=True)
    apply_parser.add_argument("--output-root", default="artifacts/perception/reviewed")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "render":
        render_review_assets(
            weak_session_root=args.weak_session_root,
            output_dir=args.output_dir,
            per_sheet=args.per_sheet,
        )
        return

    apply_review_decisions(
        weak_session_root=args.weak_session_root,
        decisions_path=args.decisions,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
