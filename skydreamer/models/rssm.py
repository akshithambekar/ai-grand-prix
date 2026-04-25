"""Dreamer-style recurrent state-space model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from skydreamer.config import ModelConfig
from skydreamer.data.schema import StepBatch, TrajectoryBatch
from skydreamer.models.actor_critic import GaussianActor
from skydreamer.models.heads import ContinuationHead, GateProgressHead, PrivilegedStateHead, RewardHead
from skydreamer.models.segmentation import SegmentationUNet


@dataclass(frozen=True)
class RSSMState:
    deter: Tensor
    stoch: Tensor
    mean: Tensor
    std: Tensor

    def feature(self) -> Tensor:
        return torch.cat([self.deter, self.stoch], dim=-1)


@dataclass(frozen=True)
class RecurrentState:
    rssm: RSSMState
    prev_action: Tensor

    def detach(self) -> "RecurrentState":
        return RecurrentState(
            rssm=RSSMState(
                deter=self.rssm.deter.detach(),
                stoch=self.rssm.stoch.detach(),
                mean=self.rssm.mean.detach(),
                std=self.rssm.std.detach(),
            ),
            prev_action=self.prev_action.detach(),
        )


@dataclass(frozen=True)
class PosteriorRollout:
    segmentation_logits: Tensor
    segmentation_probs: Tensor
    prior_mean: Tensor
    prior_std: Tensor
    posterior_mean: Tensor
    posterior_std: Tensor
    features: Tensor
    privileged_state: Tensor
    reward: Tensor
    continuation_logits: Tensor
    gate_progress_logits: Tensor
    final_state: RecurrentState


@dataclass(frozen=True)
class ImaginedRollout:
    features: Tensor
    actions: Tensor
    action_means: Tensor
    rewards: Tensor
    continuation_logits: Tensor
    values: Tensor
    entropies: Tensor
    log_probs: Tensor
    final_state: RecurrentState


class ObservationEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.seg_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
        )
        seg_dim = 64 * 8 * 8
        self.body_mlp = nn.Sequential(
            nn.Linear(config.body_rate_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )
        self.rpm_mlp = nn.Sequential(
            nn.Linear(config.motor_dim + 1, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )
        self.plan_mlp = nn.Sequential(
            nn.Linear(config.flight_plan_tokens * config.flight_plan_dim + 1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        self.fuse = nn.Sequential(
            nn.Linear(seg_dim + 32 + 32 + 64, config.obs_embed_dim),
            nn.LayerNorm(config.obs_embed_dim),
            nn.SiLU(),
            nn.Linear(config.obs_embed_dim, config.obs_embed_dim),
        )

    def forward(
        self,
        segmentation_probs: Tensor,
        body_rates: Tensor,
        motor_rpm: Tensor,
        motor_rpm_present: Tensor,
        flight_plan: Tensor,
        flight_plan_present: Tensor,
    ) -> Tensor:
        seg_features = self.seg_encoder(segmentation_probs)
        body_features = self.body_mlp(body_rates)
        rpm_features = self.rpm_mlp(torch.cat([motor_rpm, motor_rpm_present], dim=-1))
        flat_plan = flight_plan.flatten(start_dim=1)
        plan_features = self.plan_mlp(torch.cat([flat_plan, flight_plan_present], dim=-1))
        return self.fuse(torch.cat([seg_features, body_features, rpm_features, plan_features], dim=-1))


class RSSMCore(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.gru = nn.GRUCell(config.rssm.stoch_size + config.motor_dim, config.rssm.deter_size)
        self.prior = nn.Sequential(
            nn.Linear(config.rssm.deter_size, config.rssm.hidden_size),
            nn.LayerNorm(config.rssm.hidden_size),
            nn.SiLU(),
            nn.Linear(config.rssm.hidden_size, config.rssm.stoch_size * 2),
        )
        self.posterior = nn.Sequential(
            nn.Linear(config.rssm.deter_size + config.obs_embed_dim, config.rssm.hidden_size),
            nn.LayerNorm(config.rssm.hidden_size),
            nn.SiLU(),
            nn.Linear(config.rssm.hidden_size, config.rssm.stoch_size * 2),
        )

    def initial_rssm_state(self, batch_size: int, device: torch.device | str) -> RSSMState:
        deter = torch.zeros(batch_size, self.config.rssm.deter_size, device=device)
        stoch = torch.zeros(batch_size, self.config.rssm.stoch_size, device=device)
        mean = torch.zeros(batch_size, self.config.rssm.stoch_size, device=device)
        std = torch.ones(batch_size, self.config.rssm.stoch_size, device=device)
        return RSSMState(deter=deter, stoch=stoch, mean=mean, std=std)

    def step_prior(
        self,
        prev_state: RSSMState,
        prev_action: Tensor,
        *,
        sample: bool = True,
    ) -> RSSMState:
        deter = self.gru(torch.cat([prev_state.stoch, prev_action], dim=-1), prev_state.deter)
        stats = self.prior(deter)
        mean, std = self._split_stats(stats)
        stoch = mean + std * torch.randn_like(mean) if sample else mean
        return RSSMState(deter=deter, stoch=stoch, mean=mean, std=std)

    def step_posterior(
        self,
        prior_state: RSSMState,
        obs_embed: Tensor,
        *,
        sample: bool = True,
    ) -> RSSMState:
        stats = self.posterior(torch.cat([prior_state.deter, obs_embed], dim=-1))
        mean, std = self._split_stats(stats)
        stoch = mean + std * torch.randn_like(mean) if sample else mean
        return RSSMState(deter=prior_state.deter, stoch=stoch, mean=mean, std=std)

    def _split_stats(self, stats: Tensor) -> tuple[Tensor, Tensor]:
        mean, raw_std = torch.chunk(stats, 2, dim=-1)
        std = nn.functional.softplus(raw_std) + self.config.rssm.min_std
        return mean, std


class SkyDreamerWorldModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.segmentation = SegmentationUNet(config.segmentation)
        self.encoder = ObservationEncoder(config)
        self.rssm = RSSMCore(config)
        self.privileged_head = PrivilegedStateHead(config.feature_dim, config.privileged_state_dim)
        self.reward_head = RewardHead(config.feature_dim)
        self.continuation_head = ContinuationHead(config.feature_dim)
        self.gate_progress_head = GateProgressHead(config.feature_dim, config.gate_progress_classes)

    def initial_state(self, batch_size: int, device: torch.device | str) -> RecurrentState:
        return RecurrentState(
            rssm=self.rssm.initial_rssm_state(batch_size, device),
            prev_action=torch.zeros(batch_size, self.config.motor_dim, device=device),
        )

    def observe(
        self,
        batch: TrajectoryBatch,
        prev_state: RecurrentState | None = None,
    ) -> PosteriorRollout:
        batch_size, sequence_length = batch.batch_size, batch.sequence_length
        state = prev_state or self.initial_state(batch_size, batch.device)

        flat_rgb = batch.rgb.flatten(0, 1)
        seg_logits = self.segmentation(flat_rgb).view(batch_size, sequence_length, 1, 64, 64)
        seg_probs = seg_logits.sigmoid()

        prior_means: list[Tensor] = []
        prior_stds: list[Tensor] = []
        posterior_means: list[Tensor] = []
        posterior_stds: list[Tensor] = []
        features: list[Tensor] = []

        for time_index in range(sequence_length):
            obs_embed = self.encoder(
                seg_probs[:, time_index],
                batch.body_rates[:, time_index],
                batch.motor_rpm[:, time_index],
                batch.motor_rpm_present[:, time_index],
                batch.flight_plan[:, time_index],
                batch.flight_plan_present[:, time_index],
            )
            prior_state = self.rssm.step_prior(state.rssm, state.prev_action)
            posterior_state = self.rssm.step_posterior(prior_state, obs_embed)

            prior_means.append(prior_state.mean)
            prior_stds.append(prior_state.std)
            posterior_means.append(posterior_state.mean)
            posterior_stds.append(posterior_state.std)
            features.append(posterior_state.feature())

            state = RecurrentState(rssm=posterior_state, prev_action=batch.action[:, time_index])

        stacked_features = torch.stack(features, dim=1)
        return PosteriorRollout(
            segmentation_logits=seg_logits,
            segmentation_probs=seg_probs,
            prior_mean=torch.stack(prior_means, dim=1),
            prior_std=torch.stack(prior_stds, dim=1),
            posterior_mean=torch.stack(posterior_means, dim=1),
            posterior_std=torch.stack(posterior_stds, dim=1),
            features=stacked_features,
            privileged_state=self.privileged_head(stacked_features),
            reward=self.reward_head(stacked_features),
            continuation_logits=self.continuation_head(stacked_features),
            gate_progress_logits=self.gate_progress_head(stacked_features),
            final_state=state,
        )

    def observe_step(
        self,
        step: StepBatch,
        prev_state: RecurrentState | None = None,
        *,
        sample: bool = True,
    ) -> tuple[RecurrentState, dict[str, Tensor]]:
        state = prev_state or self.initial_state(step.rgb.shape[0], step.rgb.device)
        seg_logits = self.segmentation(step.rgb)
        seg_probs = seg_logits.sigmoid()
        obs_embed = self.encoder(
            seg_probs,
            step.body_rates,
            step.motor_rpm,
            step.motor_rpm_present,
            step.flight_plan,
            step.flight_plan_present,
        )
        prior_state = self.rssm.step_prior(state.rssm, state.prev_action, sample=sample)
        posterior_state = self.rssm.step_posterior(prior_state, obs_embed, sample=sample)
        next_state = RecurrentState(rssm=posterior_state, prev_action=state.prev_action)
        aux = {
            "segmentation_logits": seg_logits,
            "segmentation_probs": seg_probs,
            "feature": posterior_state.feature(),
            "prior_mean": prior_state.mean,
            "prior_std": prior_state.std,
            "posterior_mean": posterior_state.mean,
            "posterior_std": posterior_state.std,
        }
        return next_state, aux

    def imagine(
        self,
        start_state: RecurrentState,
        horizon: int,
        actor: GaussianActor,
        critic: nn.Module,
    ) -> ImaginedRollout:
        state = start_state
        features: list[Tensor] = []
        actions: list[Tensor] = []
        action_means: list[Tensor] = []
        rewards: list[Tensor] = []
        continuation_logits: list[Tensor] = []
        values: list[Tensor] = []
        entropies: list[Tensor] = []
        log_probs: list[Tensor] = []

        for _ in range(horizon):
            feature = state.rssm.feature()
            actor_output = actor(feature)
            prior_state = self.rssm.step_prior(state.rssm, state.prev_action)
            state = RecurrentState(rssm=prior_state, prev_action=actor_output.action)

            features.append(feature)
            actions.append(actor_output.action)
            action_means.append(actor_output.mean)
            rewards.append(self.reward_head(feature))
            continuation_logits.append(self.continuation_head(feature))
            values.append(critic(feature))
            entropies.append(actor_output.entropy)
            log_probs.append(actor_output.log_prob)

        return ImaginedRollout(
            features=torch.stack(features, dim=1),
            actions=torch.stack(actions, dim=1),
            action_means=torch.stack(action_means, dim=1),
            rewards=torch.stack(rewards, dim=1),
            continuation_logits=torch.stack(continuation_logits, dim=1),
            values=torch.stack(values, dim=1),
            entropies=torch.stack(entropies, dim=1),
            log_probs=torch.stack(log_probs, dim=1),
            final_state=state,
        )
