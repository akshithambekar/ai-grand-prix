"""Record synchronized simulator samples to disk for calibration and training export."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import queue

from pymavlink import mavutil

from perception.calibration import load_camera_calibration
from perception.projection import ProjectionEngine
from perception.storage import (
    append_jsonl,
    build_sample_id,
    gate_to_dict,
    projected_gate_to_dict,
    telemetry_to_dict,
    write_json,
    write_rgb_image,
)
from perception.sync import TelemetrySynchronizer
from perception.telemetry import MAVLinkTelemetryReceiver, TelemetryStateCache
from perception.vision import UDPVisionReceiver


@dataclass(frozen=True, slots=True)
class RecorderConfig:
    output_root: Path
    session_id: str
    sim_host: str
    mavlink_port: int
    vision_host: str
    vision_port: int
    calibration_path: Path | None
    max_samples: int | None
    every_nth_frame: int
    max_telemetry_age_ns: int


def record_dataset(config: RecorderConfig) -> None:
    session_root = config.output_root / config.session_id
    frames_dir = session_root / "frames"
    samples_dir = session_root / "samples"
    manifest_path = session_root / "manifest.jsonl"
    frames_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    connection = mavutil.mavlink_connection(f"udpin:{config.sim_host}:{config.mavlink_port}")
    connection.wait_heartbeat()
    cache = TelemetryStateCache()
    telemetry_receiver = MAVLinkTelemetryReceiver(connection, cache)
    vision_receiver = UDPVisionReceiver(host=config.vision_host, port=config.vision_port)
    synchronizer = TelemetrySynchronizer(cache, max_telemetry_age_ns=config.max_telemetry_age_ns)
    projection_engine = (
        ProjectionEngine(load_camera_calibration(config.calibration_path))
        if config.calibration_path is not None
        else None
    )

    telemetry_receiver.start()
    vision_receiver.start()
    recorded = 0
    seen_frames = 0

    write_json(
        session_root / "session.json",
        {
            "session_id": config.session_id,
            "created_at": datetime.now(UTC).isoformat(),
            "calibration_path": str(config.calibration_path) if config.calibration_path else None,
            "manifest_path": str(manifest_path),
        },
    )

    try:
        while config.max_samples is None or recorded < config.max_samples:
            try:
                frame = vision_receiver.get_frame(timeout=0.5)
            except queue.Empty:
                continue
            seen_frames += 1
            if seen_frames % config.every_nth_frame != 0:
                continue

            decision = synchronizer.synchronize(frame)
            if decision.sample is None:
                continue

            sample = decision.sample
            projected_gate = (
                projection_engine.project_active_gate(sample.telemetry)
                if projection_engine is not None
                else None
            )
            sample_id = build_sample_id(sample.frame.frame_id, sample.frame.timestamp_ns)
            frame_path = frames_dir / f"{sample_id}.jpg"
            metadata_path = samples_dir / f"{sample_id}.json"
            write_rgb_image(frame_path, sample.frame.rgb)

            active_gate = None
            if sample.telemetry.active_gate_index is not None:
                active_gate = sample.telemetry.gate_map.get(sample.telemetry.active_gate_index)
            payload = {
                "sample_id": sample_id,
                "session_id": config.session_id,
                "frame_id": sample.frame.frame_id,
                "frame_timestamp_ns": sample.frame.timestamp_ns,
                "telemetry_age_ns": sample.telemetry_age_ns,
                "image_path": str(frame_path),
                "telemetry": telemetry_to_dict(sample.telemetry),
                "active_gate": gate_to_dict(active_gate) if active_gate is not None else None,
                "projected_gate": projected_gate_to_dict(projected_gate),
            }
            write_json(metadata_path, payload)
            append_jsonl(
                manifest_path,
                {
                    "sample_id": sample_id,
                    "session_id": config.session_id,
                    "image_path": str(frame_path),
                    "metadata_path": str(metadata_path),
                    "projection_confidence": projected_gate.projection_confidence
                    if projected_gate is not None
                    else None,
                    "active_gate_index": sample.telemetry.active_gate_index,
                },
            )
            recorded += 1
    finally:
        vision_receiver.stop()
        telemetry_receiver.stop()
        vision_receiver.join(timeout=1.0)
        telemetry_receiver.join(timeout=1.0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="artifacts/perception/raw")
    parser.add_argument("--session-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--sim-host", default="127.0.0.1")
    parser.add_argument("--mavlink-port", type=int, default=14550)
    parser.add_argument("--vision-host", default="0.0.0.0")
    parser.add_argument("--vision-port", type=int, default=5600)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--every-nth-frame", type=int, default=1)
    parser.add_argument("--max-telemetry-age-ns", type=int, default=100_000_000)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = RecorderConfig(
        output_root=Path(args.output_root),
        session_id=args.session_id,
        sim_host=args.sim_host,
        mavlink_port=args.mavlink_port,
        vision_host=args.vision_host,
        vision_port=args.vision_port,
        calibration_path=args.calibration,
        max_samples=args.max_samples,
        every_nth_frame=args.every_nth_frame,
        max_telemetry_age_ns=args.max_telemetry_age_ns,
    )
    record_dataset(config)


if __name__ == "__main__":
    main()
