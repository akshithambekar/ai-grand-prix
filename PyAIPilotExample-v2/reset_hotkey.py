"""Send the simulator reset command when the global K key is pressed.

The ``ResetHotkey`` class can be integrated with the main controller so both use the
same MAVLink connection. Running this file directly is useful for reset testing when
no other process is bound to the simulator's MAVLink UDP port.
"""

import argparse
import threading
import time

import keyboard
from pymavlink import mavutil


MAVLINK_CMD_SIM_RESET = 31000
DEFAULT_SIM_IP = "127.0.0.1"
DEFAULT_SIM_PORT = 14550


def send_sim_reset(mavlink_connection):
    """Send one AI Grand Prix simulator reset command."""
    mavlink_connection.mav.command_long_send(
        mavlink_connection.target_system,
        mavlink_connection.target_component,
        MAVLINK_CMD_SIM_RESET,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )


class ResetHotkey:
    """Convert global K presses into reset requests consumed by the control loop."""

    def __init__(self):
        self._requested = threading.Event()
        self._hotkey = keyboard.add_hotkey(
            "k",
            self._requested.set,
            suppress=False,
            trigger_on_release=True,
        )

    def consume_request(self):
        """Return True once for each observed request state."""
        if not self._requested.is_set():
            return False
        self._requested.clear()
        return True

    def close(self):
        keyboard.remove_hotkey(self._hotkey)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Press K to send MAVLink simulator reset command 31000."
    )
    parser.add_argument("--sim-ip", default=DEFAULT_SIM_IP)
    parser.add_argument("--sim-port", type=int, default=DEFAULT_SIM_PORT)
    return parser.parse_args()


def main():
    args = _parse_args()
    connection = mavutil.mavlink_connection(
        f"udpin:{args.sim_ip}:{args.sim_port}"
    )

    print("Waiting for simulator heartbeat...", flush=True)
    connection.wait_heartbeat()
    print(
        f"Connected to system {connection.target_system}. "
        "Press K to reset; press Ctrl+C to exit.",
        flush=True,
    )

    hotkey = ResetHotkey()
    try:
        while True:
            if hotkey.consume_request():
                send_sim_reset(connection)
                print("Reset command sent.", flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Stopping reset listener.", flush=True)
    finally:
        hotkey.close()
        connection.close()


if __name__ == "__main__":
    main()

