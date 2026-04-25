"""Loss functions for the SkyDreamer scaffold."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def segmentation_loss(logits: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    probs = logits.sigmoid()
    intersection = (probs * target).sum(dim=(-3, -2, -1))
    union = probs.sum(dim=(-3, -2, -1)) + target.sum(dim=(-3, -2, -1))
    dice = 1.0 - ((2.0 * intersection + 1e-6) / (union + 1e-6)).mean()
    total = bce + dice
    return total, bce, dice


def gaussian_kl(post_mean: Tensor, post_std: Tensor, prior_mean: Tensor, prior_std: Tensor) -> Tensor:
    var_ratio = (post_std.pow(2) + (post_mean - prior_mean).pow(2)) / (prior_std.pow(2) + 1e-6)
    kl = torch.log(prior_std / (post_std + 1e-6) + 1e-6) + 0.5 * var_ratio - 0.5
    return kl.sum(dim=-1, keepdim=True)


def kl_with_free_bits(
    post_mean: Tensor,
    post_std: Tensor,
    prior_mean: Tensor,
    prior_std: Tensor,
    free_bits: float,
) -> Tensor:
    kl = gaussian_kl(post_mean, post_std, prior_mean, prior_std)
    return torch.maximum(kl, torch.full_like(kl, free_bits)).mean()


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    loss = (prediction - target).pow(2)
    if mask is None:
        return loss.mean()
    weighted = loss * mask
    return weighted.sum() / mask.sum().clamp_min(1.0)


def continuation_loss(logits: Tensor, target: Tensor) -> Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, target)


def gate_progress_loss(logits: Tensor, target: Tensor, present: Tensor) -> Tensor:
    batch_size, sequence_length, classes = logits.shape
    flattened_logits = logits.view(batch_size * sequence_length, classes)
    flattened_target = target.view(batch_size * sequence_length)
    losses = nn.functional.cross_entropy(flattened_logits, flattened_target, reduction="none")
    mask = present.view(batch_size * sequence_length)
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


def lambda_returns(
    rewards: Tensor,
    values: Tensor,
    continuation: Tensor,
    *,
    discount: float,
    lambda_: float,
) -> Tensor:
    horizon = rewards.shape[1]
    returns = torch.zeros_like(rewards)
    next_return = values[:, -1].detach()
    for time_index in reversed(range(horizon)):
        reward = rewards[:, time_index]
        cont = continuation[:, time_index] * discount
        bootstrap = values[:, time_index]
        next_return = reward + cont * ((1.0 - lambda_) * bootstrap + lambda_ * next_return)
        returns[:, time_index] = next_return
    return returns


def action_smoothness_loss(action_means: Tensor) -> Tensor:
    if action_means.shape[1] < 2:
        return torch.zeros((), device=action_means.device)
    deltas = action_means[:, 1:] - action_means[:, :-1]
    return deltas.pow(2).mean()

