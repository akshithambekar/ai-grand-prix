import math
import time
from collections.abc import Callable, Mapping, Sequence

import gymnasium as gym
import numpy as np

from controls.controller import ActionMapper
from controls.episode_manager import EpisodePhase
from controls.observation import OBSERVATION_SIZE, ObservationEncoder
from controls.reward import RewardCalculator


TERMINATED_REASONS = {
    "course_complete",
    "curriculum_complete",
    "environment_collision",
    "severe_gate_collision",
}
TRUNCATED_REASONS = {
    "gate_timeout",
    "episode_timeout",
    "gate_lost",
    "vision_stream_stall",
    "manual",
}


class AIGPEnv(gym.Env):
    """Synchronous Gym wrapper over the official simulator's asynchronous streams."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: Mapping[str, object],
        controller,
        episode_manager,
        *,
        policy_hz=10.0, # TODO: subject to change
        command_hz=50.0,
        reset_poll_s=0.01,
        reset_timeout_s=30.0,
        manual_reset_requested: Callable[[], bool] | None = None,
        close_callbacks: Sequence[Callable[[], None]] = (),
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if policy_hz <= 0 or command_hz < policy_hz:
            raise ValueError("command_hz must be at least policy_hz and both must be positive")
        commands_per_step = command_hz / policy_hz
        if not math.isclose(commands_per_step, round(commands_per_step)):
            raise ValueError("command_hz must be an integer multiple of policy_hz")
        self.data = data
        self.controller = controller
        self.episodes = episode_manager
        self.policy_period_s = 1.0 / policy_hz
        self.command_period_s = 1.0 / command_hz
        self.commands_per_step = int(round(commands_per_step))
        self.reset_poll_s = reset_poll_s
        self.reset_timeout_s = reset_timeout_s
        self.manual_reset_requested = manual_reset_requested
        self.close_callbacks = tuple(close_callbacks)
        self.clock = clock
        self.sleeper = sleeper

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -1.0,
            1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.action_mapper = ActionMapper()
        self.observations = ObservationEncoder()
        self.rewards = RewardCalculator()
        self.previous_action = np.zeros(4, dtype=np.float32)
        self._closed = False
        self._needs_reset = True
        self._reset_metrics()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        now = self.clock()
        if self.episodes.phase == EpisodePhase.IDLE:
            self.episodes.request_reset(reason="initial", now=now)
        elif self.episodes.phase == EpisodePhase.ACTIVE:
            self.episodes.request_reset(
                reason=options.get("reason", "gym_reset"),
                now=now,
            )

        deadline = now + self.reset_timeout_s
        while self.episodes.phase != EpisodePhase.ACTIVE:
            event = self.episodes.update(now=self.clock())
            if event.episode_started:
                break
            if self.clock() >= deadline:
                raise TimeoutError(
                    f"simulator reset did not become active within {self.reset_timeout_s}s; "
                    f"phase={self.episodes.phase.value}"
                )
            self.sleeper(self.reset_poll_s)

        self.previous_action = np.zeros(4, dtype=np.float32)
        self.observations.reset()
        observation = self.observations.encode(self.data, self.previous_action)
        self.rewards.reset(self.data, observation)
        self._needs_reset = False
        self._reset_metrics()
        self._episode_started_at = self.clock()
        return observation, self._build_info(None, {}, 0)

    def step(self, action):
        if self._needs_reset or self.episodes.phase != EpisodePhase.ACTIVE:
            raise RuntimeError("reset() must complete before step()")

        normalized_action = self.action_mapper.normalize(action)
        command = self.action_mapper.map(normalized_action)
        started_at = self.clock()
        command_sends = 0
        terminal_event = None

        for slot in range(self.commands_per_step):
            now = self.clock()
            if self._consume_manual_reset():
                terminal_event = self.episodes.request_reset(reason="manual", now=now)
                break

            event = self.episodes.update(now=now)
            if event.episode_ended:
                terminal_event = event
                break

            sent = self.controller.send_flight_command(
                command,
                command_allowed=event.command_allowed,
            )
            command_sends += int(bool(sent))
            slot_deadline = started_at + (slot + 1) * self.command_period_s
            self.sleeper(max(0.0, slot_deadline - self.clock()))

        if terminal_event is None:
            event = self.episodes.update(now=self.clock())
            if event.episode_ended:
                terminal_event = event

        self.previous_action = normalized_action
        observation = self.observations.encode(self.data, self.previous_action)
        reason = terminal_event.reason if terminal_event is not None else None
        reward_result = self.rewards.compute(self.data, observation, reason)
        terminated = reason in TERMINATED_REASONS
        truncated = reason in TRUNCATED_REASONS
        if reason is not None and not (terminated or truncated):
            truncated = True

        self._record_step(observation, command_sends)
        info = self._build_info(reason, reward_result.components, command_sends)
        if terminated or truncated:
            self._needs_reset = True
            info.update(self._terminal_metrics(terminal_event))
        return observation, reward_result.total, terminated, truncated, info

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.episodes.phase == EpisodePhase.ACTIVE:
            self.episodes.request_reset(reason="close", now=self.clock())
        for callback in self.close_callbacks:
            callback()

    def _consume_manual_reset(self):
        return self.manual_reset_requested is not None and self.manual_reset_requested()

    def _record_step(self, observation, command_sends):
        self._episode_steps += 1
        self._command_sends += command_sends
        if observation[0] > 0.5:
            self._detected_steps += 1
            self._center_error_sum += math.hypot(float(observation[1]), float(observation[2]))
        gate = self.data.get("gate") or {}
        frame_id = gate.get("frame_id")
        if frame_id is not None:
            self._vision_frame_ids.add(frame_id)
        status = self.data.get("race_status") or {}
        gate_index = status.get("active_gate_index")
        if gate_index is not None:
            self._highest_gate_index = max(self._highest_gate_index, int(gate_index))
        collision = self.data.get("collision") or {}
        collision_sequence = int(collision.get("sequence", 0) or 0)
        if collision_sequence > self._last_collision_sequence:
            self._collision_count += 1
            self._last_collision_sequence = collision_sequence

    def _build_info(self, reason, reward_components, command_sends):
        gate = self.data.get("gate") or {}
        status = self.data.get("race_status") or {}
        return {
            "episode_id": self.episodes.episode_id,
            "active_gate_index": status.get("active_gate_index"),
            "gates_passed": self.episodes.gates_passed,
            "track_id": gate.get("track_id"),
            "detected": bool(gate.get("detected", False)),
            "termination_reason": reason,
            "reward_components": dict(reward_components),
            "command_sends": command_sends,
        }

    def _terminal_metrics(self, event):
        duration = max(self.clock() - self._episode_started_at, 1e-9)
        detected = max(self._detected_steps, 1)
        return {
            "episode_duration_s": duration,
            "episode_steps": self._episode_steps,
            "highest_gate_index": self._highest_gate_index,
            "detection_rate": self._detected_steps / max(self._episode_steps, 1),
            "mean_center_error": self._center_error_sum / detected,
            "control_rate_hz": self._command_sends / duration,
            "vision_rate_hz": len(self._vision_frame_ids) / duration,
            "collision_count": self._collision_count,
            "reset_reason": event.reason if event is not None else None,
        }

    def _reset_metrics(self):
        self._episode_started_at = self.clock()
        self._episode_steps = 0
        self._detected_steps = 0
        self._center_error_sum = 0.0
        self._command_sends = 0
        self._vision_frame_ids = set()
        self._highest_gate_index = 0
        collision = self.data.get("collision") or {}
        self._last_collision_sequence = int(collision.get("sequence", 0) or 0)
        self._collision_count = 0
