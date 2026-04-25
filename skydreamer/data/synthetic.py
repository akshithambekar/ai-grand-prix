"""Synthetic trajectory generation for smoke checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from skydreamer.data.schema import TrajectoryBatch


@dataclass
class SyntheticBatchGenerator:
    image_height: int = 96
    image_width: int = 96
    gate_progress_classes: int = 32

    def make_batch(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        include_rpm: bool = True,
        include_flight_plan: bool = True,
        device: torch.device | str = "cpu",
    ) -> TrajectoryBatch:
        rgb = torch.rand(batch_size, sequence_length, 3, self.image_height, self.image_width, device=device)
        seg_target = self._make_segmentation_targets(batch_size, sequence_length, device=device)
        body_rates = torch.randn(batch_size, sequence_length, 3, device=device) * 0.2
        action = torch.sigmoid(torch.randn(batch_size, sequence_length, 4, device=device))
        reward = torch.randn(batch_size, sequence_length, 1, device=device) * 0.1
        continuation = torch.ones(batch_size, sequence_length, 1, device=device)
        privileged_state = torch.randn(batch_size, sequence_length, 13, device=device) * 0.1

        motor_rpm = None
        motor_rpm_present = None
        if include_rpm:
            motor_rpm = torch.rand(batch_size, sequence_length, 4, device=device)
            motor_rpm_present = torch.ones(batch_size, sequence_length, 1, device=device)

        flight_plan = None
        flight_plan_present = None
        if include_flight_plan:
            flight_plan = torch.randn(batch_size, sequence_length, 3, 4, device=device)
            flight_plan_present = torch.ones(batch_size, sequence_length, 1, device=device)

        gate_progress_index = torch.randint(
            low=0,
            high=self.gate_progress_classes,
            size=(batch_size, sequence_length),
            device=device,
        )
        gate_progress_present = torch.ones(batch_size, sequence_length, 1, device=device)

        return TrajectoryBatch.from_optional_inputs(
            rgb=rgb,
            seg_target=seg_target,
            body_rates=body_rates,
            motor_rpm=motor_rpm,
            motor_rpm_present=motor_rpm_present,
            flight_plan=flight_plan,
            flight_plan_present=flight_plan_present,
            action=action,
            reward=reward,
            continuation=continuation,
            privileged_state=privileged_state,
            gate_progress_index=gate_progress_index,
            gate_progress_present=gate_progress_present,
        )

    def _make_segmentation_targets(
        self,
        batch_size: int,
        sequence_length: int,
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        masks = torch.zeros(batch_size, sequence_length, 1, 64, 64, device=device)
        for batch_index in range(batch_size):
            for time_index in range(sequence_length):
                center_x = torch.randint(12, 52, size=(1,), device=device).item()
                center_y = torch.randint(12, 52, size=(1,), device=device).item()
                half_size = torch.randint(4, 12, size=(1,), device=device).item()
                x0 = max(0, center_x - half_size)
                x1 = min(64, center_x + half_size)
                y0 = max(0, center_y - half_size)
                y1 = min(64, center_y + half_size)
                masks[batch_index, time_index, 0, y0:y1, x0:x1] = 1.0
        return masks

