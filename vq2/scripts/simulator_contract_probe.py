"""Inventory VQ2 streams and optionally apply one bounded command pulse."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

from pymavlink import mavutil


VQ2_ROOT = Path(__file__).resolve().parent.parent
if str(VQ2_ROOT) not in sys.path:
    sys.path.insert(0, str(VQ2_ROOT))

from scripts.evidence import (  # noqa: E402
    CommandSender,
    EvidenceWriter,
    FrameRecorder,
    ManualControlConfig,
    TelemetryRecorder,
    create_run_dir,
    map_manual_keys,
    write_summary,
)


EXPECTED_STREAMS = ("HEARTBEAT", "HIGHRES_IMU", "ACTUATOR_OUTPUT_STATUS", "ENCAPSULATED_DATA")
DISABLED_STREAMS = ("ATTITUDE", "LOCAL_POSITION_NED", "ODOMETRY")
IMU_FIELDS = ("xacc", "yacc", "zacc", "xgyro", "ygyro", "zgyro")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save all FPV/MAVLink evidence and inventory the VQ2 simulator contract. "
            "Without --execute, no arm, reset, or flight command is sent."
        )
    )
    parser.add_argument("--sim-ip", default="127.0.0.1")
    parser.add_argument("--sim-port", type=int, default=14550)
    parser.add_argument("--vision-ip", default="0.0.0.0")
    parser.add_argument("--vision-port", type=int, default=5600)
    parser.add_argument("--passive-seconds", type=float, default=8.0)
    parser.add_argument("--reset-settle-seconds", type=float, default=4.0)
    parser.add_argument("--phase-seconds", type=float, default=0.5)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--mode", choices=("attitude", "velocity", "motor"), default="attitude")
    parser.add_argument("--hover-thrust", type=float, default=0.42)
    parser.add_argument("--thrust-step", type=float, default=0.03)
    parser.add_argument("--rate", type=float, default=0.08)
    parser.add_argument("--speed", type=float, default=0.75)
    parser.add_argument(
        "--motor-hover",
        type=float,
        help="Required for an active raw-motor probe; no universal safe value is known.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=VQ2_ROOT / "artifacts" / "contract"
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if min(args.passive_seconds, args.reset_settle_seconds, args.phase_seconds, args.control_hz) <= 0:
        parser.error("durations and control rate must be positive")
    if args.execute and args.mode == "motor" and args.motor_hover is None:
        parser.error("active motor mode requires an explicitly measured --motor-hover value")
    return args


def sample_imu(telemetry: TelemetryRecorder) -> dict[str, float] | None:
    message = telemetry.latest("HIGHRES_IMU")
    if message is None:
        return None
    try:
        return {field: float(message[field]) for field in IMU_FIELDS}
    except (KeyError, TypeError, ValueError):
        return None


def phase_imu_summary(samples: list[dict[str, float]]) -> dict[str, float | int]:
    result: dict[str, float | int] = {"samples": len(samples)}
    for field in IMU_FIELDS:
        values = [sample[field] for sample in samples]
        if values:
            result[f"{field}_mean"] = statistics.fmean(values)
            result[f"{field}_min"] = min(values)
            result[f"{field}_max"] = max(values)
    return result


def run_phase(
    name: str,
    keys: set[str],
    duration_s: float,
    control_hz: float,
    config: ManualControlConfig,
    mode: str,
    sender: CommandSender,
    telemetry: TelemetryRecorder,
    writer: EvidenceWriter,
) -> list[dict[str, float]]:
    command = map_manual_keys(keys, mode, config)
    writer.write("probe_phase", name=name, keys=sorted(keys), command=asdict(command))
    print(f"Active phase: {name} ({duration_s:.1f}s)", flush=True)
    samples = []
    deadline = time.monotonic() + duration_s
    next_tick = time.monotonic()
    tick = 0
    timesync_interval = max(1, round(control_hz / 10.0))
    while time.monotonic() < deadline:
        if tick % timesync_interval == 0:
            client_time_ns = sender.send_timesync()
            writer.write("timesync_request", client_time_ns=client_time_ns)
        sender.send(command)
        writer.write("command", source="contract_probe", phase=name, command=asdict(command))
        sample = sample_imu(telemetry)
        if sample is not None:
            samples.append(sample)
        next_tick += 1.0 / control_hz
        tick += 1
        time.sleep(max(0.0, next_tick - time.monotonic()))
    return samples


def passive_inventory(
    duration_s: float, sender: CommandSender, writer: EvidenceWriter
) -> None:
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        client_time_ns = sender.send_timesync()
        writer.write("timesync_request", client_time_ns=client_time_ns)
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def main() -> None:
    args = parse_args()
    run_dir = create_run_dir(args.output_root, "contract")
    writer = EvidenceWriter(run_dir)
    connection = mavutil.mavlink_connection(f"udpin:{args.sim_ip}:{args.sim_port}")
    telemetry = TelemetryRecorder(connection, writer)
    frames = FrameRecorder(args.vision_ip, args.vision_port, writer)
    sender = CommandSender(connection)
    phase_samples: dict[str, list[dict[str, float]]] = {}
    connected = False
    writer.write("session_start", tool="simulator_contract_probe", arguments=vars(args))
    print(f"Evidence directory: {run_dir}", flush=True)
    print("Waiting for simulator heartbeat...", flush=True)
    try:
        heartbeat = connection.wait_heartbeat(timeout=15)
        if heartbeat is None:
            raise RuntimeError("no simulator heartbeat received within 15 seconds")
        connected = True
        writer.write(
            "connected",
            target_system=connection.target_system,
            target_component=connection.target_component,
        )
        telemetry.start()
        frames.start()
        print(f"Passive stream inventory: {args.passive_seconds:.1f}s", flush=True)
        passive_inventory(args.passive_seconds, sender, writer)

        if args.execute:
            print("Resetting simulator before the bounded active probe...", flush=True)
            sender.reset()
            writer.write("sim_reset", source="contract_probe")
            time.sleep(args.reset_settle_seconds)
            sender.arm()
            writer.write("arm", source="contract_probe")
            config = ManualControlConfig(
                hover_thrust=args.hover_thrust,
                thrust_step=args.thrust_step,
                roll_rate=args.rate,
                pitch_rate=args.rate,
                forward_speed=args.speed,
                motor_hover=args.motor_hover if args.motor_hover is not None else 0.0,
            )
            pulse_keys = {"q"} if args.mode == "motor" else {"w"}
            for name, keys in (
                ("neutral_before", set()),
                ("bounded_pulse", pulse_keys),
                ("neutral_after", set()),
            ):
                phase_samples[name] = run_phase(
                    name,
                    keys,
                    args.phase_seconds,
                    args.control_hz,
                    config,
                    args.mode,
                    sender,
                    telemetry,
                    writer,
                )
            sender.reset()
            writer.write("sim_reset", source="contract_probe_complete")
    except KeyboardInterrupt:
        writer.write("session_interrupt")
        if connected and args.execute:
            sender.reset()
    finally:
        frames.stop()
        telemetry.stop()
        message_counts = dict(sorted(telemetry.message_counts.items()))
        track_data = telemetry.latest("TRACK_DATA")
        summary = {
            "tool": "simulator_contract_probe",
            "execute": args.execute,
            "control_mode": args.mode if args.execute else None,
            "message_counts": message_counts,
            "expected_streams": {
                name: message_counts.get(name, 0) > 0 for name in EXPECTED_STREAMS
            },
            "disabled_streams_observed": {
                name: message_counts.get(name, 0) for name in DISABLED_STREAMS
            },
            "track_data": track_data,
            "frames": frames.summary(),
            "phase_imu": {
                name: phase_imu_summary(samples) for name, samples in phase_samples.items()
            },
            "event_counts": dict(writer.counts),
        }
        write_summary(run_dir, summary)
        writer.close()
        connection.close()
        print(f"Contract summary: {run_dir / 'summary.json'}", flush=True)
        print(
            f"Frames={summary['frames']['frame_count']} "
            f"cyan_present={summary['frames']['cyan_present']} "
            f"messages={message_counts}",
            flush=True,
        )


if __name__ == "__main__":
    main()
