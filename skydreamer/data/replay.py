"""Simple contiguous-chunk replay sampling."""

from __future__ import annotations

import random
from dataclasses import replace

from skydreamer.data.schema import TrajectoryBatch


class SequenceReplayBuffer:
    """Stores full trajectory batches and samples contiguous subsequences."""

    def __init__(self, sequence_length: int = 32) -> None:
        self.sequence_length = sequence_length
        self._episodes: list[TrajectoryBatch] = []

    def add(self, episode: TrajectoryBatch) -> None:
        if episode.sequence_length < self.sequence_length:
            msg = (
                f"episode length {episode.sequence_length} shorter than configured "
                f"chunk length {self.sequence_length}"
            )
            raise ValueError(msg)
        self._episodes.append(episode)

    def __len__(self) -> int:
        return len(self._episodes)

    def sample(self) -> TrajectoryBatch:
        if not self._episodes:
            raise ValueError("replay buffer is empty")
        episode = random.choice(self._episodes)
        max_start = episode.sequence_length - self.sequence_length
        start = random.randint(0, max_start)
        end = start + self.sequence_length
        return replace(
            episode,
            rgb=episode.rgb[:, start:end],
            seg_target=episode.seg_target[:, start:end],
            body_rates=episode.body_rates[:, start:end],
            motor_rpm=episode.motor_rpm[:, start:end],
            motor_rpm_present=episode.motor_rpm_present[:, start:end],
            flight_plan=episode.flight_plan[:, start:end],
            flight_plan_present=episode.flight_plan_present[:, start:end],
            action=episode.action[:, start:end],
            reward=episode.reward[:, start:end],
            continuation=episode.continuation[:, start:end],
            privileged_state=episode.privileged_state[:, start:end],
            gate_progress_index=episode.gate_progress_index[:, start:end],
            gate_progress_present=episode.gate_progress_present[:, start:end],
        )

