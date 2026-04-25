"""Pilot command types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PilotCommand:
    timestamp: float
    throttle: float
    roll: float
    pitch: float
    yaw: float

    def clamped(self, limit: "CommandLimits") -> "PilotCommand":
        return PilotCommand(
            timestamp=self.timestamp,
            throttle=_clamp(self.throttle, limit.min_throttle, limit.max_throttle),
            roll=_clamp(self.roll, -limit.max_roll, limit.max_roll),
            pitch=_clamp(self.pitch, -limit.max_pitch, limit.max_pitch),
            yaw=_clamp(self.yaw, -limit.max_yaw, limit.max_yaw),
        )


@dataclass(frozen=True)
class CommandLimits:
    min_throttle: float
    max_throttle: float
    max_roll: float
    max_pitch: float
    max_yaw: float


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))

