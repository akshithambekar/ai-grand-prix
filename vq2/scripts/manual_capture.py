"""Fly with WASD/QE while recording synchronized VQ2 evidence."""

from __future__ import annotations

import argparse
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
    FlightCommand,
    FrameRecorder,
    KeyPoller,
    ManualControlConfig,
    TelemetryRecorder,
    create_run_dir,
    map_manual_keys,
    update_throttle,
    write_summary,
)


CONTROL_KEYS = frozenset("wasdqe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture all FPV frames and MAVLink messages. Hold W/A/S/D for horizontal "
            "motion, Q for up, E for down, R to reset, and Escape to finish."
        )
    )
    parser.add_argument("--sim-ip", default="127.0.0.1")
    parser.add_argument("--sim-port", type=int, default=14550)
    parser.add_argument("--vision-ip", default="0.0.0.0")
    parser.add_argument("--vision-port", type=int, default=5600)
    parser.add_argument("--mode", choices=("attitude", "velocity", "motor"), default="attitude")
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--reset-settle-seconds", type=float, default=4.0)
    parser.add_argument("--initial-thrust", type=float, default=0.0)
    parser.add_argument("--thrust-ramp-per-second", type=float, default=0.20)
    parser.add_argument("--max-thrust", type=float, default=0.50)
    parser.add_argument("--roll-rate", type=float, default=0.08)
    parser.add_argument("--pitch-rate", type=float, default=0.08)
    parser.add_argument("--forward-speed", type=float, default=1.0)
    parser.add_argument("--lateral-speed", type=float, default=1.0)
    parser.add_argument("--vertical-speed", type=float, default=0.6)
    parser.add_argument(
        "--motor-hover",
        type=float,
        help="Required for active raw-motor control; measure it with the contract probe first.",
    )
    parser.add_argument("--motor-thrust-step", type=float, default=0.08)
    parser.add_argument("--motor-tilt-step", type=float, default=0.04)
    parser.add_argument("--output-root", type=Path, default=VQ2_ROOT / "artifacts" / "manual")
    parser.add_argument(
        "--reset-before-run", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--reset-on-exit", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if min(args.control_hz, args.reset_settle_seconds, args.thrust_ramp_per_second) <= 0:
        parser.error("control rate and reset settle time must be positive")
    if not 0.0 <= args.initial_thrust <= args.max_thrust <= 1.0:
        parser.error("thrust must satisfy 0 <= initial <= maximum <= 1")
    if args.execute and args.mode == "motor" and args.motor_hover is None:
        parser.error("active motor mode requires --motor-hover")
    return args


def main() -> None:
    args = parse_args()
    run_dir = create_run_dir(args.output_root, "manual")
    writer = EvidenceWriter(run_dir)
    connection = mavutil.mavlink_connection(f"udpin:{args.sim_ip}:{args.sim_port}")
    telemetry = TelemetryRecorder(connection, writer)
    frames = FrameRecorder(args.vision_ip, args.vision_port, writer)
    sender = CommandSender(connection)
    keys = KeyPoller()
    config = ManualControlConfig(
        hover_thrust=args.initial_thrust,
        thrust_step=0.0,
        roll_rate=args.roll_rate,
        pitch_rate=args.pitch_rate,
        forward_speed=args.forward_speed,
        lateral_speed=args.lateral_speed,
        vertical_speed=args.vertical_speed,
        motor_hover=args.motor_hover if args.motor_hover is not None else 0.0,
        motor_thrust_step=args.motor_thrust_step,
        motor_tilt_step=args.motor_tilt_step,
    )
    writer.write("session_start", tool="manual_capture", arguments=vars(args), config=asdict(config))
    print(f"Evidence directory: {run_dir}", flush=True)
    print("Waiting for simulator heartbeat...", flush=True)
    reset_count = 0
    commands_sent = 0
    previous_keys: set[str] = set()
    manual_thrust = args.initial_thrust
    connected = False
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
        command_allowed_at = time.monotonic()
        arm_sent = False
        if args.execute and args.reset_before_run:
            sender.reset()
            reset_count += 1
            writer.write("sim_reset", source="manual_capture_start")
            command_allowed_at += args.reset_settle_seconds
            print(
                f"Reset sent; controls unlock in {args.reset_settle_seconds:.1f}s.", flush=True
            )
        elif args.execute:
            sender.arm()
            arm_sent = True
            writer.write("arm", source="manual_capture_start")

        print(
            "Hold W/A/S/D to move, Q up, E down, R reset, Escape finish. "
            + (f"LIVE {args.mode} commands enabled." if args.execute else "CAPTURE-ONLY mode."),
            flush=True,
        )
        next_tick = time.monotonic()
        previous_tick = next_tick
        tick = 0
        timesync_interval = max(1, round(args.control_hz / 10.0))
        while True:
            now = time.monotonic()
            elapsed_s = now - previous_tick
            previous_tick = now
            if tick % timesync_interval == 0:
                client_time_ns = sender.send_timesync()
                writer.write("timesync_request", client_time_ns=client_time_ns)
            pressed = keys.pressed()
            reset_pressed = "r" in pressed and "r" not in previous_keys
            if pressed != previous_keys:
                writer.write("keys", pressed=sorted(pressed))
                previous_keys = pressed
            if "escape" in pressed:
                break
            if reset_pressed and args.execute:
                sender.reset()
                reset_count += 1
                manual_thrust = args.initial_thrust
                arm_sent = False
                command_allowed_at = now + args.reset_settle_seconds
                writer.write("sim_reset", source="manual_key")
                print("Reset sent; commands temporarily locked.", flush=True)

            if args.execute and now >= command_allowed_at and not arm_sent:
                sender.arm()
                arm_sent = True
                writer.write("arm", source="manual_capture")
                print("Armed; manual commands live.", flush=True)

            sent = bool(args.execute and arm_sent and now >= command_allowed_at)
            if args.mode == "attitude":
                if sent:
                    manual_thrust = update_throttle(
                        manual_thrust,
                        pressed,
                        elapsed_s,
                        args.thrust_ramp_per_second,
                        args.max_thrust,
                    )
                rates = map_manual_keys(pressed & {"w", "a", "s", "d"}, args.mode, config)
                command = FlightCommand("attitude", (*rates.values[:3], manual_thrust))
            else:
                command = map_manual_keys(pressed & CONTROL_KEYS, args.mode, config)
            if sent:
                sender.send(command)
                commands_sent += 1
            writer.write(
                "command",
                source="manual_capture",
                keys=sorted(pressed & CONTROL_KEYS),
                sent=sent,
                command=asdict(command),
                manual_thrust=manual_thrust if args.mode == "attitude" else None,
            )
            next_tick += 1.0 / args.control_hz
            tick += 1
            time.sleep(max(0.0, next_tick - time.monotonic()))
    except KeyboardInterrupt:
        writer.write("session_interrupt")
    finally:
        if connected and args.execute and args.reset_on_exit:
            sender.reset()
            reset_count += 1
            writer.write("sim_reset", source="manual_capture_exit")
        frames.stop()
        telemetry.stop()
        summary = {
            "tool": "manual_capture",
            "execute": args.execute,
            "control_mode": args.mode,
            "commands_sent": commands_sent,
            "reset_count": reset_count,
            "message_counts": dict(sorted(telemetry.message_counts.items())),
            "frames": frames.summary(),
            "event_counts": dict(writer.counts),
        }
        write_summary(run_dir, summary)
        writer.close()
        connection.close()
        print(f"Capture summary: {run_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
