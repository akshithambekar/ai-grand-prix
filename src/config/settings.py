"""Configuration for the conservative VQ1 stack."""

from __future__ import annotations

from dataclasses import dataclass, field

from control.commands import CommandLimits


@dataclass(frozen=True)
class PerceptionSettings:
    max_processing_width: int = 1280
    min_gate_confidence: float = 0.60
    high_gate_confidence: float = 0.85


@dataclass(frozen=True)
class ControlSettings:
    command_rate_hz: float = 60.0
    hover_throttle: float = 0.35
    approach_pitch: float = 0.15
    search_throttle: float = 0.30
    search_yaw: float = 0.12
    failsafe_throttle: float = 0.25
    min_command_confidence: float = 0.60
    max_perception_age_s: float = 0.15
    command_smoothing_alpha: float = 0.25
    roll_gain: float = 0.40
    yaw_gain: float = 0.35
    vertical_gain: float = 0.25
    limits: CommandLimits = field(
        default_factory=lambda: CommandLimits(
            min_throttle=0.0,
            max_throttle=0.55,
            max_roll=0.35,
            max_pitch=0.35,
            max_yaw=0.35,
        )
    )


@dataclass(frozen=True)
class VQ1Settings:
    perception: PerceptionSettings = field(default_factory=PerceptionSettings)
    control: ControlSettings = field(default_factory=ControlSettings)

