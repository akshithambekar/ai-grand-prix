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
    logits: Tensor
    probs: Tensor

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
                logits=self.rssm.logits.detach(),
                probs=self.rssm.probs.detach(),
            ),
            prev_action=self.prev_action.detach(),
        )


@dataclass(frozen=True)
class PosteriorRollout:
    segmentation_logits: Tensor
    segmentation_probs: Tensor
    prior_logits: Tensor
    prior_probs: Tensor
    posterior_logits: Tensor
    posterior_probs: Tensor
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
        self.gru = nn.GRUCell(config.rssm.stoch_dim + config.motor_dim, config.rssm.deter_size)
        self.prior = nn.Sequential(
            nn.Linear(config.rssm.deter_size, config.rssm.hidden_size),
            nn.LayerNorm(config.rssm.hidden_size),
            nn.SiLU(),
            nn.Linear(config.rssm.hidden_size, config.rssm.stoch_dim),
        )
        self.posterior = nn.Sequential(
            nn.Linear(config.rssm.deter_size + config.obs_embed_dim, config.rssm.hidden_size),
            nn.LayerNorm(config.rssm.hidden_size),
            nn.SiLU(),
            nn.Linear(config.rssm.hidden_size, config.rssm.stoch_dim),
        )

    def initial_rssm_state(self, batch_size: int, device: torch.device | str) -> RSSMState:
        deter = torch.zeros(batch_size, self.config.rssm.deter_size, device=device)
        logits = torch.zeros(
            batch_size,
            self.config.rssm.stoch_size,
            self.config.rssm.stoch_classes,
            device=device,
        )
        probs = logits.softmax(dim=-1)
        stoch = self._sample_discrete(logits, sample=False)
        return RSSMState(deter=deter, stoch=stoch, logits=logits, probs=probs)

    def step_prior(
        self,
        prev_state: RSSMState,
        prev_action: Tensor,
        *,
        sample: bool = True,
    ) -> RSSMState:
        deter = self.gru(torch.cat([prev_state.stoch, prev_action], dim=-1), prev_state.deter)
        logits = self._reshape_logits(self.prior(deter))
        probs = logits.softmax(dim=-1)
        stoch = self._sample_discrete(logits, sample=sample)
        return RSSMState(deter=deter, stoch=stoch, logits=logits, probs=probs)

    def step_posterior(
        self,
        prior_state: RSSMState,
        obs_embed: Tensor,
        *,
        sample: bool = True,
    ) -> RSSMState:
        logits = self._reshape_logits(self.posterior(torch.cat([prior_state.deter, obs_embed], dim=-1)))
        probs = logits.softmax(dim=-1)
        stoch = self._sample_discrete(logits, sample=sample)
        return RSSMState(deter=prior_state.deter, stoch=stoch, logits=logits, probs=probs)

    def _reshape_logits(self, logits: Tensor) -> Tensor:
        return logits.view(-1, self.config.rssm.stoch_size, self.config.rssm.stoch_classes)

    def _sample_discrete(self, logits: Tensor, *, sample: bool) -> Tensor:
        if sample:
            stoch = nn.functional.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        else:
            indices = logits.argmax(dim=-1)
            stoch = nn.functional.one_hot(indices, num_classes=self.config.rssm.stoch_classes).to(
                logits.dtype
            )
        return stoch.flatten(start_dim=1)


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

        prior_logits: list[Tensor] = []
        prior_probs: list[Tensor] = []
        posterior_logits: list[Tensor] = []
        posterior_probs: list[Tensor] = []
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

            prior_logits.append(prior_state.logits)
            prior_probs.append(prior_state.probs)
            posterior_logits.append(posterior_state.logits)
            posterior_probs.append(posterior_state.probs)
            features.append(posterior_state.feature())

            state = RecurrentState(rssm=posterior_state, prev_action=batch.action[:, time_index])

        stacked_features = torch.stack(features, dim=1)
        return PosteriorRollout(
            segmentation_logits=seg_logits,
            segmentation_probs=seg_probs,
            prior_logits=torch.stack(prior_logits, dim=1),
            prior_probs=torch.stack(prior_probs, dim=1),
            posterior_logits=torch.stack(posterior_logits, dim=1),
            posterior_probs=torch.stack(posterior_probs, dim=1),
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
            "prior_logits": prior_state.logits,
            "prior_probs": prior_state.probs,
            "posterior_logits": posterior_state.logits,
            "posterior_probs": posterior_state.probs,
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
            prior_state = self.rssm.step_prior(state.rssm, actor_output.action)
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
