"""Trainer and smoke checks for the SkyDreamer scaffold."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from skydreamer.config import SkyDreamerConfig
from skydreamer.data.schema import TrajectoryBatch
from skydreamer.models.policy import SkyDreamerPolicy
from skydreamer.models.rssm import ImaginedRollout, PosteriorRollout
from skydreamer.train.losses import (
    action_smoothness_loss,
    continuation_loss,
    gate_progress_loss,
    kl_with_free_bits,
    lambda_returns,
    masked_mse,
    segmentation_loss,
)


@dataclass(frozen=True)
class TrainStepOutput:
    total_loss: float
    metrics: dict[str, float]
    step: int


class SkyDreamerTrainer:
    def __init__(
        self,
        config: SkyDreamerConfig | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config or SkyDreamerConfig()
        self.device = torch.device(device)
        self.policy = SkyDreamerPolicy(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        self.step_count = 0

    def train_step(self, batch: TrajectoryBatch) -> TrainStepOutput:
        self.policy.train()
        batch = batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)

        posterior = self.policy.world_model.observe(batch)
        imagined = self.policy.world_model.imagine(
            start_state=posterior.final_state.detach(),
            horizon=self.config.training.imagination_horizon,
            actor=self.policy.actor,
            critic=self.policy.critic,
        )

        total_loss, metrics = self.compute_losses(batch, posterior, imagined)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.step_count += 1
        metric_values = {name: float(value.detach().cpu()) for name, value in metrics.items()}
        return TrainStepOutput(
            total_loss=float(total_loss.detach().cpu()),
            metrics=metric_values,
            step=self.step_count,
        )

    def compute_losses(
        self,
        batch: TrajectoryBatch,
        posterior: PosteriorRollout,
        imagined: ImaginedRollout,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        cfg = self.config.training
        seg_total, seg_bce, seg_dice = segmentation_loss(
            posterior.segmentation_logits, batch.seg_target
        )
        kl_loss = kl_with_free_bits(
            posterior.posterior_mean,
            posterior.posterior_std,
            posterior.prior_mean,
            posterior.prior_std,
            cfg.free_bits,
        )
        privileged_loss = masked_mse(posterior.privileged_state, batch.privileged_state)
        reward_loss = masked_mse(posterior.reward, batch.reward)
        cont_loss = continuation_loss(posterior.continuation_logits, batch.continuation)
        gate_loss = gate_progress_loss(
            posterior.gate_progress_logits,
            batch.gate_progress_index,
            batch.gate_progress_present.squeeze(-1),
        )

        imagined_cont = posterior.continuation_logits.new_full(
            posterior.continuation_logits[:, : cfg.imagination_horizon].shape, 1.0
        )
        imagined_rewards = imagined.rewards
        imagined_values = imagined.values
        lambda_ret = lambda_returns(
            imagined_rewards,
            imagined_values,
            imagined_cont,
            discount=cfg.discount,
            lambda_=cfg.lambda_,
        )
        critic_loss = torch.nn.functional.huber_loss(imagined_values, lambda_ret.detach())
        actor_objective = (lambda_ret - cfg.entropy_scale * imagined.entropies).mean()
        actor_loss = -actor_objective
        smooth_loss = action_smoothness_loss(imagined.action_means)

        total_loss = (
            cfg.segmentation_scale * seg_total
            + kl_loss
            + cfg.reconstruction_scale * privileged_loss
            + cfg.reward_scale * reward_loss
            + cfg.continuation_scale * cont_loss
            + cfg.gate_progress_scale * gate_loss
            + cfg.critic_scale * critic_loss
            + cfg.actor_scale * actor_loss
            + cfg.smoothness_scale * smooth_loss
        )

        metrics = {
            "segmentation_total": seg_total,
            "segmentation_bce": seg_bce,
            "segmentation_dice": seg_dice,
            "kl": kl_loss,
            "privileged_state": privileged_loss,
            "reward": reward_loss,
            "continuation": cont_loss,
            "gate_progress": gate_loss,
            "critic": critic_loss,
            "actor": actor_loss,
            "smoothness": smooth_loss,
            "lambda_return_mean": lambda_ret.mean(),
        }
        return total_loss, metrics

