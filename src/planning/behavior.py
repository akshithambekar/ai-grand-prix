"""High-level behavior selection skeleton."""

from __future__ import annotations

from state.race_state import RaceMode, RaceState


class BehaviorPlanner:
    def choose_mode(self, state: RaceState) -> RaceMode:
        return state.mode

