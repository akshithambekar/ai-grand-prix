"""Actor-critic modules for direct motor control."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


@dataclass(frozen=True)
class ActorOutput:
    action: Tensor
    mean: Tensor
    std: Tensor
    log_prob: Tensor
    entropy: Tensor


class GaussianActor(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.backbone = _mlp(feature_dim, hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, features: Tensor, *, deterministic: bool = False) -> ActorOutput:
        hidden = self.backbone(features)
        mean = self.mean_head(hidden)
        log_std = torch.clamp(self.log_std_head(hidden), min=-5.0, max=2.0)
        std = log_std.exp()

        if deterministic:
            pre_squash = mean
        else:
            noise = torch.randn_like(mean)
            pre_squash = mean + std * noise

        squashed = torch.tanh(pre_squash)
        action = 0.5 * (squashed + 1.0)

        normal_log_prob = -0.5 * (
            ((pre_squash - mean) / (std + 1e-6)) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        normal_log_prob = normal_log_prob.sum(dim=-1, keepdim=True)
        squash_correction = torch.log(1.0 - squashed.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        log_prob = normal_log_prob - squash_correction
        entropy = (0.5 + 0.5 * math.log(2.0 * math.pi) + log_std).sum(dim=-1, keepdim=True)

        mean_action = 0.5 * (torch.tanh(mean) + 1.0)
        return ActorOutput(
            action=action,
            mean=mean_action,
            std=std,
            log_prob=log_prob,
            entropy=entropy,
        )


class Critic(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp(feature_dim, hidden_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)

