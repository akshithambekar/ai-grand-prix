"""End-to-end RGB-to-motor policy wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from skydreamer.config import SkyDreamerConfig
from skydreamer.data.schema import StepBatch
from skydreamer.models.actor_critic import Critic, GaussianActor
from skydreamer.models.rssm import RecurrentState, SkyDreamerWorldModel


class SkyDreamerPolicy(nn.Module):
    def __init__(self, config: SkyDreamerConfig | None = None) -> None:
        super().__init__()
        self.config = config or SkyDreamerConfig()
        model_config = self.config.model
        self.world_model = SkyDreamerWorldModel(model_config)
        self.actor = GaussianActor(
            feature_dim=model_config.feature_dim,
            hidden_dim=model_config.actor_hidden_dim,
            action_dim=model_config.motor_dim,
        )
        self.critic = Critic(
            feature_dim=model_config.feature_dim,
            hidden_dim=model_config.critic_hidden_dim,
        )

    def initial_state(self, batch_size: int, device: torch.device | str = "cpu") -> RecurrentState:
        return self.world_model.initial_state(batch_size, device)

    def act(
        self,
        rgb: Tensor,
        body_rates: Tensor,
        motor_rpm: Tensor | None = None,
        flight_plan: Tensor | None = None,
        prev_state: RecurrentState | None = None,
        *,
        deterministic: bool = True,
        motor_rpm_present: Tensor | None = None,
        flight_plan_present: Tensor | None = None,
    ) -> tuple[Tensor, RecurrentState, dict[str, Tensor]]:
        step = StepBatch.from_optional_inputs(
            rgb=rgb,
            body_rates=body_rates,
            motor_rpm=motor_rpm,
            flight_plan=flight_plan,
            motor_rpm_present=motor_rpm_present,
            flight_plan_present=flight_plan_present,
        )
        next_state, aux = self.world_model.observe_step(
            step,
            prev_state,
            sample=not deterministic,
        )
        actor_output = self.actor(aux["feature"], deterministic=deterministic)
        updated_state = RecurrentState(rssm=next_state.rssm, prev_action=actor_output.action)
        aux.update(
            {
                "action_mean": actor_output.mean,
                "action_std": actor_output.std,
                "log_prob": actor_output.log_prob,
                "entropy": actor_output.entropy,
                "value": self.critic(aux["feature"]),
            }
        )
        return actor_output.action, updated_state, aux


@dataclass
class InferenceSession:
    """Carries recurrent state across deterministic action calls."""

    policy: SkyDreamerPolicy
    state: RecurrentState

    @classmethod
    def create(
        cls,
        policy: SkyDreamerPolicy,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
    ) -> "InferenceSession":
        return cls(policy=policy, state=policy.initial_state(batch_size=batch_size, device=device))

    @torch.no_grad()
    def act(
        self,
        rgb: Tensor,
        body_rates: Tensor,
        motor_rpm: Tensor | None = None,
        flight_plan: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        action, next_state, aux = self.policy.act(
            rgb=rgb,
            body_rates=body_rates,
            motor_rpm=motor_rpm,
            flight_plan=flight_plan,
            prev_state=self.state,
            deterministic=True,
        )
        self.state = next_state.detach()
        return action, aux
