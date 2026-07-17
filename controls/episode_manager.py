"""Episode lifecycle and reset conditions for simulator training."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Optional


class EpisodePhase(str, Enum):
    IDLE = "IDLE"
    RESET_REQUESTED = "RESET_REQUESTED"
    COUNTDOWN = "COUNTDOWN"
    WAITING_FOR_ARM = "WAITING_FOR_ARM"
    AWAIT_FRESH_FRAME = "AWAIT_FRESH_FRAME"
    ACTIVE = "ACTIVE"


@dataclass(frozen=True)
class EpisodeConfig:
    gate_timeout_s: float = 15.0
    vision_timeout_s: float = 1.0
    no_detection_timeout_s: float = 2.0
    pass_grace_s: float = 0.5
    episode_timeout_s: float = 120.0
    arm_retry_s: float = 1.0
    post_reset_command_delay_s: float = 4.0
    target_gate_count: Optional[int] = None


@dataclass(frozen=True)
class EpisodeEvent:
    phase: EpisodePhase
    command_allowed: bool
    episode_started: bool = False
    episode_ended: bool = False
    reset_requested: bool = False
    reason: Optional[str] = None
    episode_id: int = 0
    gates_passed: int = 0
    result: Optional[Mapping[str, object]] = None


class EpisodeManager:
    """Drive episode boundaries from observable simulator state.

    ``data`` is the shared snapshot populated by the MAVLink and vision receivers.
    The manager never sends flight-control commands itself; callers should send a
    command only when ``EpisodeEvent.command_allowed`` is true.
    """

    def __init__(
        self,
        data: Mapping[str, object],
        send_reset: Callable[[], None],
        send_arm: Callable[[], None],
        config: EpisodeConfig | None = None,
    ):
        self.data = data
        self.send_reset = send_reset
        self.send_arm = send_arm
        self.config = config or EpisodeConfig()

        self.phase = EpisodePhase.IDLE
        self.episode_id = 0
        self.gates_passed = 0
        self.last_result = None

        self._reset_sent_at = None
        self._reference_sim_ms = None
        self._reference_start_ms = None
        self._baseline_frame_id = None
        self._last_frame_id = None
        self._last_frame_at = None
        self._last_arm_at = None
        self._episode_started_at = None
        self._gate_started_at = None
        self._gate_changed_at = None
        self._current_gate = None
        self._no_detection_since = None
        self._collision_sequence = 0

    @property
    def command_allowed(self):
        return self.phase == EpisodePhase.ACTIVE

    def request_reset(self, reason="manual", now=None):
        """Start a reset handshake and return the resulting event."""
        now = self._now() if now is None else now
        was_active = self.phase == EpisodePhase.ACTIVE
        self._reset_sent_at = now
        self._reference_sim_ms, self._reference_start_ms = self._race_reference()
        self._baseline_frame_id = self._frame_id()
        self._last_frame_id = self._baseline_frame_id
        self._last_frame_at = None
        self._last_arm_at = None
        self._episode_started_at = None
        self._gate_started_at = None
        self._gate_changed_at = None
        self._current_gate = None
        self._no_detection_since = None
        self._collision_sequence = self._collision_seq()
        self.phase = EpisodePhase.RESET_REQUESTED
        self.send_reset()
        return self._event(
            reset_requested=True,
            episode_ended=was_active,
            reason=reason if was_active else None,
        )

    def update(self, now=None):
        """Advance the state machine using current shared-state snapshots."""
        now = self._now() if now is None else now
        if self.phase == EpisodePhase.IDLE:
            return self._event()

        if self.phase == EpisodePhase.RESET_REQUESTED:
            self._observe_reset_status()
            return self._event()

        if self.phase == EpisodePhase.COUNTDOWN:
            if self._countdown_finished():
                self.phase = EpisodePhase.WAITING_FOR_ARM
            return self._event()

        if self.phase == EpisodePhase.WAITING_FOR_ARM:
            if now - self._reset_sent_at < self.config.post_reset_command_delay_s:
                return self._event()
            if not self._armed():
                if self._last_arm_at is None or now - self._last_arm_at >= self.config.arm_retry_s:
                    self.send_arm()
                    self._last_arm_at = now
                return self._event()
            self.phase = EpisodePhase.AWAIT_FRESH_FRAME
            return self._event()

        if self.phase == EpisodePhase.AWAIT_FRESH_FRAME:
            if self._has_fresh_frame():
                self._begin_episode(now)
                return self._event(episode_started=True)
            return self._event()

        return self._update_active(now)

    def _observe_reset_status(self):
        status = self._race_status()
        if status is None or not self._received_after_reset(status):
            return

        start_ms = status.get("race_start_boot_time_ms", -1)
        sim_ms = status.get("sim_boot_time_ms", -1)
        if not self._is_new_reset_epoch(status):
            return
        if start_ms >= 0 and sim_ms < start_ms:
            self.phase = EpisodePhase.COUNTDOWN
        elif start_ms >= 0 and sim_ms >= start_ms:
            self.phase = EpisodePhase.WAITING_FOR_ARM

    def _update_active(self, now):
        status = self._race_status() or {}
        gate = status.get("active_gate_index")
        if gate is not None and self._current_gate is not None and gate > self._current_gate:
            self.gates_passed += gate - self._current_gate
            self._current_gate = gate
            self._gate_started_at = now
            self._gate_changed_at = now
            self._no_detection_since = None

        reason = self._termination_reason(now)
        if reason is None:
            return self._event()

        result = {
            "reason": reason,
            "episode_id": self.episode_id,
            "gates_passed": self.gates_passed,
            "duration_s": now - self._episode_started_at,
        }
        self.last_result = result
        event = self.request_reset(reason=reason, now=now)
        return EpisodeEvent(
            phase=event.phase,
            command_allowed=False,
            episode_ended=True,
            reset_requested=True,
            reason=reason,
            episode_id=self.episode_id,
            gates_passed=self.gates_passed,
            result=result,
        )

    def _termination_reason(self, now):
        status = self._race_status() or {}
        if status.get("race_finish_time_ns", -1) >= 0:
            return "course_complete"

        if (
            self.config.target_gate_count is not None
            and self.gates_passed >= self.config.target_gate_count
        ):
            return "curriculum_complete"

        if self._episode_started_at is not None and now - self._episode_started_at >= self.config.episode_timeout_s:
            return "episode_timeout"

        if self._gate_started_at is not None and now - self._gate_started_at >= self.config.gate_timeout_s:
            return "gate_timeout"

        collision = self.data.get("collision") or {}
        if collision.get("sequence", 0) > self._collision_sequence:
            collision_id = collision.get("collision_id")
            threat = collision.get("threat_level", 0)
            if collision_id == 1002:
                return "environment_collision"
            if collision_id == 1001 and threat >= 2:
                return "severe_gate_collision"
            self._collision_sequence = collision.get("sequence", self._collision_sequence)

        gate = self.data.get("gate") or {}
        frame_id = gate.get("frame_id")
        if frame_id is not None and frame_id != self._last_frame_id:
            self._last_frame_id = frame_id
            self._last_frame_at = now
            if gate.get("detected", False):
                self._no_detection_since = None
            elif self._no_detection_since is None:
                self._no_detection_since = now

        if self._last_frame_at is not None and now - self._last_frame_at >= self.config.vision_timeout_s:
            return "vision_stream_stall"

        if (
            self._no_detection_since is not None
            and now - self._no_detection_since >= self.config.no_detection_timeout_s
            and (
                self._gate_changed_at is None
                or now - self._gate_changed_at >= self.config.pass_grace_s
            )
        ):
            return "gate_lost"
        return None

    def _begin_episode(self, now):
        status = self._race_status() or {}
        self.phase = EpisodePhase.ACTIVE
        self.episode_id += 1
        self.gates_passed = 0
        self._episode_started_at = now
        self._gate_started_at = now
        self._gate_changed_at = None
        self._last_frame_at = now
        self._current_gate = status.get("active_gate_index", 0)
        self._collision_sequence = self._collision_seq()
        self._no_detection_since = None

    def _has_fresh_frame(self):
        gate = self.data.get("gate") or {}
        frame_id = gate.get("frame_id")
        if frame_id is None:
            return False
        if self._baseline_frame_id is not None and frame_id <= self._baseline_frame_id:
            return False
        status = self._race_status() or {}
        start_ms = status.get("race_start_boot_time_ms", -1)
        timestamp_s = gate.get("timestamp_s")
        return start_ms < 0 or timestamp_s is None or timestamp_s * 1000 >= start_ms

    def _is_new_reset_epoch(self, status):
        start_ms = status.get("race_start_boot_time_ms", -1)
        sim_ms = status.get("sim_boot_time_ms", -1)
        if self._reference_sim_ms is not None and sim_ms < self._reference_sim_ms:
            return False
        return start_ms != self._reference_start_ms or (
            start_ms >= 0 and sim_ms < start_ms
        )

    def _countdown_finished(self):
        status = self._race_status() or {}
        start_ms = status.get("race_start_boot_time_ms", -1)
        return start_ms >= 0 and status.get("sim_boot_time_ms", -1) >= start_ms

    def _received_after_reset(self, status):
        received_at = status.get("received_at_s")
        return received_at is not None and received_at >= self._reset_sent_at

    def _race_status(self):
        status = self.data.get("race_status")
        return status if isinstance(status, Mapping) else None

    def _race_reference(self):
        status = self._race_status() or {}
        return status.get("sim_boot_time_ms"), status.get("race_start_boot_time_ms")

    def _frame_id(self):
        gate = self.data.get("gate") or {}
        return gate.get("frame_id")

    def _collision_seq(self):
        collision = self.data.get("collision") or {}
        return collision.get("sequence", 0)

    def _armed(self):
        return bool(self.data.get("armed", False))

    @staticmethod
    def _now():
        import time

        return time.monotonic()

    def _event(self, **kwargs):
        return EpisodeEvent(
            phase=self.phase,
            command_allowed=self.command_allowed,
            episode_id=self.episode_id,
            gates_passed=self.gates_passed,
            result=self.last_result,
            **kwargs,
        )
