"""Formal data schema for SkyDreamer trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _require_rank(tensor: Tensor, rank: int, name: str) -> None:
    if tensor.ndim != rank:
        msg = f"{name} must have rank {rank}, got {tensor.ndim}"
        raise ValueError(msg)


def _require_last_dim(tensor: Tensor, expected: int, name: str) -> None:
    if tensor.shape[-1] != expected:
        msg = f"{name} must have last dimension {expected}, got {tensor.shape[-1]}"
        raise ValueError(msg)


def _fill_optional(
    tensor: Tensor | None,
    present: Tensor | None,
    *,
    shape: tuple[int, ...],
    present_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    if tensor is None:
        return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(
            present_shape, device=device, dtype=torch.float32
        )
    if present is None:
        present = torch.ones(present_shape, device=tensor.device, dtype=torch.float32)
    return tensor, present


@dataclass(frozen=True)
class StepBatch:
    """Single-timestep inference batch."""

    rgb: Tensor
    body_rates: Tensor
    motor_rpm: Tensor
    motor_rpm_present: Tensor
    flight_plan: Tensor
    flight_plan_present: Tensor

    @classmethod
    def from_optional_inputs(
        cls,
        rgb: Tensor,
        body_rates: Tensor,
        motor_rpm: Tensor | None = None,
        flight_plan: Tensor | None = None,
        *,
        motor_rpm_present: Tensor | None = None,
        flight_plan_present: Tensor | None = None,
    ) -> "StepBatch":
        if rgb.ndim != 4:
            msg = f"rgb must be [B, 3, H, W], got {tuple(rgb.shape)}"
            raise ValueError(msg)
        if body_rates.ndim != 2 or body_rates.shape[-1] != 3:
            msg = f"body_rates must be [B, 3], got {tuple(body_rates.shape)}"
            raise ValueError(msg)
        batch_size = rgb.shape[0]
        device = rgb.device
        rpm, rpm_present = _fill_optional(
            motor_rpm,
            motor_rpm_present,
            shape=(batch_size, 4),
            present_shape=(batch_size, 1),
            device=device,
        )
        plan, plan_present = _fill_optional(
            flight_plan,
            flight_plan_present,
            shape=(batch_size, 3, 4),
            present_shape=(batch_size, 1),
            device=device,
        )
        return cls(
            rgb=rgb,
            body_rates=body_rates,
            motor_rpm=rpm,
            motor_rpm_present=rpm_present,
            flight_plan=plan,
            flight_plan_present=plan_present,
        )


@dataclass(frozen=True)
class TrajectoryBatch:
    """Sequence training batch with normalized optional inputs."""

    rgb: Tensor
    seg_target: Tensor
    body_rates: Tensor
    motor_rpm: Tensor
    motor_rpm_present: Tensor
    flight_plan: Tensor
    flight_plan_present: Tensor
    action: Tensor
    reward: Tensor
    continuation: Tensor
    privileged_state: Tensor
    gate_progress_index: Tensor
    gate_progress_present: Tensor

    def __post_init__(self) -> None:
        _require_rank(self.rgb, 5, "rgb")
        _require_rank(self.seg_target, 5, "seg_target")
        _require_rank(self.body_rates, 3, "body_rates")
        _require_rank(self.motor_rpm, 3, "motor_rpm")
        _require_rank(self.motor_rpm_present, 3, "motor_rpm_present")
        _require_rank(self.flight_plan, 4, "flight_plan")
        _require_rank(self.flight_plan_present, 3, "flight_plan_present")
        _require_rank(self.action, 3, "action")
        _require_rank(self.reward, 3, "reward")
        _require_rank(self.continuation, 3, "continuation")
        _require_rank(self.privileged_state, 3, "privileged_state")
        _require_rank(self.gate_progress_index, 2, "gate_progress_index")
        _require_rank(self.gate_progress_present, 3, "gate_progress_present")

        _require_last_dim(self.body_rates, 3, "body_rates")
        _require_last_dim(self.motor_rpm, 4, "motor_rpm")
        _require_last_dim(self.motor_rpm_present, 1, "motor_rpm_present")
        _require_last_dim(self.flight_plan, 4, "flight_plan")
        _require_last_dim(self.flight_plan_present, 1, "flight_plan_present")
        _require_last_dim(self.action, 4, "action")
        _require_last_dim(self.reward, 1, "reward")
        _require_last_dim(self.continuation, 1, "continuation")
        _require_last_dim(self.privileged_state, 13, "privileged_state")
        _require_last_dim(self.gate_progress_present, 1, "gate_progress_present")

    @property
    def batch_size(self) -> int:
        return self.rgb.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.rgb.shape[1]

    @property
    def device(self) -> torch.device:
        return self.rgb.device

    def to(self, device: torch.device | str) -> "TrajectoryBatch":
        return TrajectoryBatch(
            rgb=self.rgb.to(device),
            seg_target=self.seg_target.to(device),
            body_rates=self.body_rates.to(device),
            motor_rpm=self.motor_rpm.to(device),
            motor_rpm_present=self.motor_rpm_present.to(device),
            flight_plan=self.flight_plan.to(device),
            flight_plan_present=self.flight_plan_present.to(device),
            action=self.action.to(device),
            reward=self.reward.to(device),
            continuation=self.continuation.to(device),
            privileged_state=self.privileged_state.to(device),
            gate_progress_index=self.gate_progress_index.to(device),
            gate_progress_present=self.gate_progress_present.to(device),
        )

    @classmethod
    def from_optional_inputs(
        cls,
        *,
        rgb: Tensor,
        seg_target: Tensor,
        body_rates: Tensor,
        action: Tensor,
        reward: Tensor,
        continuation: Tensor,
        privileged_state: Tensor,
        motor_rpm: Tensor | None = None,
        motor_rpm_present: Tensor | None = None,
        flight_plan: Tensor | None = None,
        flight_plan_present: Tensor | None = None,
        gate_progress_index: Tensor | None = None,
        gate_progress_present: Tensor | None = None,
    ) -> "TrajectoryBatch":
        if rgb.ndim != 5:
            msg = f"rgb must be [B, T, 3, H, W], got {tuple(rgb.shape)}"
            raise ValueError(msg)
        batch_size, sequence_length = rgb.shape[:2]
        device = rgb.device
        rpm, rpm_present = _fill_optional(
            motor_rpm,
            motor_rpm_present,
            shape=(batch_size, sequence_length, 4),
            present_shape=(batch_size, sequence_length, 1),
            device=device,
        )
        plan, plan_present = _fill_optional(
            flight_plan,
            flight_plan_present,
            shape=(batch_size, sequence_length, 3, 4),
            present_shape=(batch_size, sequence_length, 1),
            device=device,
        )
        if gate_progress_index is None:
            gate_progress_index = torch.zeros(
                (batch_size, sequence_length), device=device, dtype=torch.long
            )
        if gate_progress_present is None:
            gate_progress_present = torch.zeros(
                (batch_size, sequence_length, 1), device=device, dtype=torch.float32
            )
        return cls(
            rgb=rgb,
            seg_target=seg_target,
            body_rates=body_rates,
            motor_rpm=rpm,
            motor_rpm_present=rpm_present,
            flight_plan=plan,
            flight_plan_present=plan_present,
            action=action,
            reward=reward,
            continuation=continuation,
            privileged_state=privileged_state,
            gate_progress_index=gate_progress_index,
            gate_progress_present=gate_progress_present,
        )
