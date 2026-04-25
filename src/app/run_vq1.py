"""VQ1 runner skeleton.

The simulator API has not been released yet, so this module wires the intended
runtime loop without binding to a concrete client implementation.
"""

from __future__ import annotations

from config.settings import VQ1Settings
from control.safety import SafetyMonitor
from control.visual_servo import VisualServoController
from perception.gate_detector import GateDetector
from perception.tracking import GateTracker
from planning.search import SearchBehavior
from sim.api_client import SimulatorClient
from state.race_state import RaceStateMachine


def run_vq1(client: SimulatorClient, settings: VQ1Settings | None = None) -> None:
    """Run the reliability-first VQ1 control loop."""
    settings = settings or VQ1Settings()
    detector = GateDetector(settings.perception)
    tracker = GateTracker(settings.perception)
    race_state = RaceStateMachine()
    controller = VisualServoController(settings.control)
    search = SearchBehavior(settings.control)
    safety = SafetyMonitor(settings.control)

    client.connect()
    try:
        while True:
            telemetry = client.read_telemetry()
            frame = client.read_frame()
            observations = detector.detect(frame)
            gate = tracker.update(observations, telemetry)
            state = race_state.update(telemetry=telemetry, gate=gate, command=None)

            if safety.should_failsafe(telemetry=telemetry, gate=gate):
                command = safety.failsafe_command(telemetry.timestamp)
            elif gate is None:
                command = search.command(telemetry, state)
            else:
                command = controller.compute_command(telemetry, gate, state)

            client.send_command(command)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit("Provide a concrete SimulatorClient once the API is released.")

