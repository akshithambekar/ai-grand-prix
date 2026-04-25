"""Configuration for the SkyDreamer scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SegmentationConfig:
    input_channels: int = 3
    base_channels: int = 32
    output_size: int = 64


@dataclass(frozen=True)
class RSSMConfig:
    deter_size: int = 512
    stoch_size: int = 32
    hidden_size: int = 512
    min_std: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    rssm: RSSMConfig = field(default_factory=RSSMConfig)
    body_rate_dim: int = 3
    motor_dim: int = 4
    flight_plan_tokens: int = 3
    flight_plan_dim: int = 4
    privileged_state_dim: int = 13
    gate_progress_classes: int = 32
    obs_embed_dim: int = 256
    feature_dim: int = 544
    actor_hidden_dim: int = 512
    critic_hidden_dim: int = 512


@dataclass(frozen=True)
class TrainingConfig:
    sequence_length: int = 32
    batch_size: int = 4
    imagination_horizon: int = 8
    discount: float = 0.99
    lambda_: float = 0.95
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    free_bits: float = 1.0
    entropy_scale: float = 1e-3
    smoothness_scale: float = 1e-2
    segmentation_scale: float = 1.0
    reconstruction_scale: float = 1.0
    reward_scale: float = 1.0
    continuation_scale: float = 1.0
    critic_scale: float = 1.0
    actor_scale: float = 1.0
    gate_progress_scale: float = 0.25


@dataclass(frozen=True)
class SkyDreamerConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

