import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from pymavlink import mavutil


MAVLINK_CMD_SIM_RESET = 31000
RATES_ATTITUDE_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE


@dataclass(frozen=True)
class FlightCommand:
    roll_rate: float
    pitch_rate: float
    yaw_rate: float
    thrust: float


class ActionMapper:
    """Map normalized PPO actions to conservative absolute flight commands."""

    ROLL_LIMITS = (-0.20, 0.20)
    PITCH_LIMITS = (-0.20, 0.15)
    YAW_LIMITS = (-0.15, 0.15)
    THRUST_LIMITS = (0.23, 0.30)

    def map(self, action: Iterable[float]) -> FlightCommand:
        normalized = self.normalize(action)
        return FlightCommand(
            roll_rate=self._map_rate(normalized[0], self.ROLL_LIMITS),
            pitch_rate=self._map_rate(normalized[1], self.PITCH_LIMITS),
            yaw_rate=self._map_rate(normalized[2], self.YAW_LIMITS),
            thrust=self._map_linear(normalized[3], self.THRUST_LIMITS),
        )

    def action_for_command(self, command: FlightCommand) -> np.ndarray:
        """Inverse mapping used by deterministic live probes."""
        return np.asarray([
            self._normalize_rate(command.roll_rate, self.ROLL_LIMITS),
            self._normalize_rate(command.pitch_rate, self.PITCH_LIMITS),
            self._normalize_rate(command.yaw_rate, self.YAW_LIMITS),
            self._normalize_linear(command.thrust, self.THRUST_LIMITS),
        ], dtype=np.float32)

    @staticmethod
    def normalize(action: Iterable[float]) -> np.ndarray:
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (4,):
            raise ValueError(f"action must have shape (4,), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("action must contain only finite values")
        return np.clip(values, -1.0, 1.0)

    @staticmethod
    def _map_rate(value: float, limits: tuple[float, float]) -> float:
        low, high = limits
        return float(value * (high if value >= 0 else -low))

    @staticmethod
    def _normalize_rate(value: float, limits: tuple[float, float]) -> float:
        low, high = limits
        clamped = min(max(float(value), low), high)
        return clamped / (high if clamped >= 0 else -low)

    @staticmethod
    def _map_linear(value: float, limits: tuple[float, float]) -> float:
        low, high = limits
        return float(low + 0.5 * (value + 1.0) * (high - low))

    @staticmethod
    def _normalize_linear(value: float, limits: tuple[float, float]) -> float:
        low, high = limits
        clamped = min(max(float(value), low), high)
        return 2.0 * (clamped - low) / (high - low) - 1.0


class Controller:
    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms
        self.action_mapper = ActionMapper()

    def send_normalized_action(self, action, command_allowed):
        command = self.action_mapper.map(action)
        return self.send_flight_command(command, command_allowed)

    def send_flight_command(self, command: FlightCommand, command_allowed: bool):
        if not command_allowed:
            return False
        values = (
            command.roll_rate,
            command.pitch_rate,
            command.yaw_rate,
            command.thrust,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("flight command must contain only finite values")

        now_ms = int(time.time() * 1000)
        self.sim_conn.mav.set_attitude_target_send(
            now_ms - self.system_boot_ms,
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            RATES_ATTITUDE_MASK,
            [1, 0, 0, 0],
            command.roll_rate,
            command.pitch_rate,
            command.yaw_rate,
            min(max(command.thrust, 0.0), 1.0),
        )
        return True

    def arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,
            0, 0, 0, 0, 0, 0,
        )

    def send_sim_reset_command(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            MAVLINK_CMD_SIM_RESET,
            0,
            0, 0, 0, 0, 0, 0, 0,
        )
