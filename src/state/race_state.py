"""Race state machine for conservative VQ1 behavior."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from control.commands import PilotCommand
from perception.observations import GateObservation
from sim.types import Telemetry


class RaceMode(Enum):
    INITIALIZING = "initializing"
    SEARCHING = "searching"
    APPROACHING_GATE = "approaching_gate"
    PASSING_GATE = "passing_gate"
    REACQUIRING = "reacquiring"
    FINISHED = "finished"
    FAILSAFE = "failsafe"


@dataclass(frozen=True)
class RaceState:
    mode: RaceMode
    last_gate_seen_at: float | None = None


class RaceStateMachine:
    def __init__(self) -> None:
        self.state = RaceState(mode=RaceMode.INITIALIZING)

    def update(
        self,
        telemetry: Telemetry,
        gate: GateObservation | None,
        command: PilotCommand | None,
    ) -> RaceState:
        _ = command
        if gate is None:
            self.state = RaceState(
                mode=RaceMode.SEARCHING,
                last_gate_seen_at=self.state.last_gate_seen_at,
            )
        else:
            self.state = RaceState(
                mode=RaceMode.APPROACHING_GATE,
                last_gate_seen_at=telemetry.timestamp,
            )
        return self.state

