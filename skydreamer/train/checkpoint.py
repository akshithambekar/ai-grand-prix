"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import Optimizer

from skydreamer.models.policy import SkyDreamerPolicy


def save_checkpoint(
    path: str | Path,
    *,
    policy: SkyDreamerPolicy,
    optimizer: Optimizer,
    step: int,
) -> None:
    checkpoint = {
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    torch.save(checkpoint, Path(path))


def load_checkpoint(
    path: str | Path,
    *,
    policy: SkyDreamerPolicy,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> int:
    checkpoint = torch.load(Path(path), map_location=map_location)
    policy.load_state_dict(checkpoint["policy"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])

