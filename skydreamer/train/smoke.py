"""Smoke checks for the SkyDreamer scaffold."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from skydreamer.config import SkyDreamerConfig
from skydreamer.data.replay import SequenceReplayBuffer
from skydreamer.data.synthetic import SyntheticBatchGenerator
from skydreamer.models.policy import InferenceSession, SkyDreamerPolicy
from skydreamer.train.checkpoint import load_checkpoint, save_checkpoint
from skydreamer.train.trainer import SkyDreamerTrainer


def run_smoke_checks(device: torch.device | str = "cpu") -> dict[str, float]:
    device = torch.device(device)
    torch.manual_seed(0)
    config = SkyDreamerConfig()
    generator = SyntheticBatchGenerator(gate_progress_classes=config.model.gate_progress_classes)

    trainer = SkyDreamerTrainer(config, device=device)
    batch = generator.make_batch(
        batch_size=2,
        sequence_length=config.training.sequence_length,
        include_rpm=True,
        include_flight_plan=True,
        device=device,
    )
    output = trainer.train_step(batch)
    posterior = trainer.policy.world_model.observe(batch)
    imagined = trainer.policy.world_model.imagine(
        start_state=posterior.final_state.detach(),
        horizon=config.training.imagination_horizon,
        actor=trainer.policy.actor,
        critic=trainer.policy.critic,
    )

    if posterior.features.shape != (
        batch.batch_size,
        batch.sequence_length,
        config.model.feature_dim,
    ):
        raise AssertionError("posterior rollout features have the wrong shape")
    if imagined.actions.shape != (
        batch.batch_size,
        config.training.imagination_horizon,
        config.model.motor_dim,
    ):
        raise AssertionError("imagined rollout actions have the wrong shape")

    replay = SequenceReplayBuffer(sequence_length=config.training.sequence_length)
    replay.add(batch)
    replay_sample = replay.sample()
    if replay_sample.rgb.shape != batch.rgb.shape:
        raise AssertionError("replay sample shape mismatch")

    policy = trainer.policy.eval()
    rgb = batch.rgb[:, 0]
    body_rates = batch.body_rates[:, 0]
    action, next_state, aux = policy.act(
        rgb=rgb,
        body_rates=body_rates,
        motor_rpm=batch.motor_rpm[:, 0],
        flight_plan=batch.flight_plan[:, 0],
        prev_state=policy.initial_state(batch.batch_size, device),
        deterministic=True,
    )

    if not torch.all((action >= 0.0) & (action <= 1.0)):
        raise AssertionError("actor outputs must stay within [0, 1]")

    session = InferenceSession.create(policy, batch_size=1, device=device)
    session_action, _ = session.act(batch.rgb[:1, 0], batch.body_rates[:1, 0])
    if not torch.all((session_action >= 0.0) & (session_action <= 1.0)):
        raise AssertionError("inference session action must stay within [0, 1]")

    missing_rpm_batch = generator.make_batch(
        batch_size=2,
        sequence_length=config.training.sequence_length,
        include_rpm=False,
        include_flight_plan=True,
        device=device,
    )
    missing_rpm_output = trainer.train_step(missing_rpm_batch)

    missing_plan_batch = generator.make_batch(
        batch_size=2,
        sequence_length=config.training.sequence_length,
        include_rpm=True,
        include_flight_plan=False,
        device=device,
    )
    missing_plan_output = trainer.train_step(missing_plan_batch)

    overfit_batch = generator.make_batch(
        batch_size=1,
        sequence_length=config.training.sequence_length,
        include_rpm=True,
        include_flight_plan=True,
        device=device,
    )
    overfit_start = trainer.train_step(overfit_batch).total_loss
    overfit_end = overfit_start
    for _ in range(4):
        torch.manual_seed(0)
        overfit_end = trainer.train_step(overfit_batch).total_loss
    if overfit_end > overfit_start:
        raise AssertionError("tiny-batch overfit check did not reduce loss")

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "skydreamer.pt"
        save_checkpoint(checkpoint_path, policy=trainer.policy, optimizer=trainer.optimizer, step=trainer.step_count)

        restored = SkyDreamerTrainer(config, device=device)
        restored_step = load_checkpoint(
            checkpoint_path,
            policy=restored.policy,
            optimizer=restored.optimizer,
            map_location=device,
        )
        if restored_step != trainer.step_count:
            raise AssertionError("checkpoint step mismatch")

        trainer.policy.eval()
        restored.policy.eval()
        original_action, _, _ = trainer.policy.act(
            rgb=rgb,
            body_rates=body_rates,
            motor_rpm=batch.motor_rpm[:, 0],
            flight_plan=batch.flight_plan[:, 0],
            prev_state=trainer.policy.initial_state(batch.batch_size, device),
            deterministic=True,
        )
        restored_action, _, _ = restored.policy.act(
            rgb=rgb,
            body_rates=body_rates,
            motor_rpm=batch.motor_rpm[:, 0],
            flight_plan=batch.flight_plan[:, 0],
            prev_state=restored.policy.initial_state(batch.batch_size, device),
            deterministic=True,
        )
        if not torch.allclose(original_action, restored_action, atol=1e-5):
            raise AssertionError("checkpoint restore changed deterministic actions")

    results = {
        "train_step_loss": output.total_loss,
        "missing_rpm_loss": missing_rpm_output.total_loss,
        "missing_plan_loss": missing_plan_output.total_loss,
        "overfit_start": overfit_start,
        "overfit_end": overfit_end,
        "action_mean": float(action.mean().detach().cpu()),
        "next_state_norm": float(next_state.rssm.feature().norm().detach().cpu()),
        "value_mean": float(aux["value"].mean().detach().cpu()),
    }
    return results
