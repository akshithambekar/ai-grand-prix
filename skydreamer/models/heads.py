"""Prediction heads on top of latent features."""

from __future__ import annotations

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


class PrivilegedStateHead(nn.Module):
    def __init__(self, feature_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = _mlp(feature_dim, feature_dim, output_dim)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class RewardHead(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = _mlp(feature_dim, feature_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class ContinuationHead(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = _mlp(feature_dim, feature_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class GateProgressHead(nn.Module):
    def __init__(self, feature_dim: int, classes: int) -> None:
        super().__init__()
        self.net = _mlp(feature_dim, feature_dim, classes)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)

