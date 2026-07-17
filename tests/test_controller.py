import unittest

import numpy as np

from controls.controller import ActionMapper, Controller, FlightCommand


class FakeMAV:
    def __init__(self):
        self.attitude_calls = []

    def set_attitude_target_send(self, *args):
        self.attitude_calls.append(args)


class FakeConnection:
    def __init__(self):
        self.target_system = 1
        self.target_component = 2
        self.mav = FakeMAV()


class ActionMapperTests(unittest.TestCase):
    def test_zero_action_is_hover(self):
        command = ActionMapper().map(np.zeros(4, dtype=np.float32))
        self.assertAlmostEqual(command.roll_rate, 0.0)
        self.assertAlmostEqual(command.pitch_rate, 0.0)
        self.assertAlmostEqual(command.yaw_rate, 0.0)
        self.assertAlmostEqual(command.thrust, 0.265)

    def test_action_bounds_map_to_command_bounds(self):
        mapper = ActionMapper()
        low = mapper.map([-1, -1, -1, -1])
        high = mapper.map([1, 1, 1, 1])
        for actual, expected in zip(low.__dict__.values(), (-0.20, -0.20, -0.15, 0.23)):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(high.__dict__.values(), (0.20, 0.15, 0.15, 0.30)):
            self.assertAlmostEqual(actual, expected)

    def test_invalid_action_is_rejected(self):
        with self.assertRaises(ValueError):
            ActionMapper().map([0, 0, float("nan"), 0])
        with self.assertRaises(ValueError):
            ActionMapper().map([0, 0, 0])

    def test_command_guard_blocks_transmission(self):
        connection = FakeConnection()
        controller = Controller(connection, {}, 0)
        command = FlightCommand(0.0, 0.0, 0.0, 0.265)

        self.assertFalse(controller.send_flight_command(command, False))
        self.assertEqual(connection.mav.attitude_calls, [])
        self.assertTrue(controller.send_flight_command(command, True))
        self.assertEqual(len(connection.mav.attitude_calls), 1)


if __name__ == "__main__":
    unittest.main()
