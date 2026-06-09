from __future__ import annotations

import struct
import unittest

from perception.telemetry import parse_race_status_payload, parse_track_data_payload


class TelemetryParsingTests(unittest.TestCase):
    def test_parse_track_data_payload_decodes_gate_map(self) -> None:
        payload = struct.pack(
            "<H" "Hfffffffff",
            1,
            7,
            10.0,
            20.0,
            -3.0,
            1.0,
            0.1,
            0.2,
            0.3,
            5.5,
            2.25,
        )

        gate_map = parse_track_data_payload(payload)

        self.assertEqual(set(gate_map), {7})
        gate = gate_map[7]
        self.assertEqual(gate.center_position_m, (10.0, 20.0, -3.0))
        self.assertAlmostEqual(gate.orientation_xyzw[0], 0.1)
        self.assertAlmostEqual(gate.orientation_xyzw[1], 0.2)
        self.assertAlmostEqual(gate.orientation_xyzw[2], 0.3)
        self.assertAlmostEqual(gate.orientation_xyzw[3], 1.0)
        self.assertEqual(gate.width_m, 5.5)
        self.assertEqual(gate.height_m, 2.25)

    def test_parse_race_status_payload_normalizes_negative_sentinels(self) -> None:
        payload = struct.pack("<BQqqIq", 1, 42, -1, -1, 3, -1)

        status = parse_race_status_payload(payload)

        self.assertEqual(status.sim_boot_time_ms, 42)
        self.assertIsNone(status.race_start_boot_time_ms)
        self.assertIsNone(status.race_finish_time_ns)
        self.assertEqual(status.active_gate_index, 3)
        self.assertIsNone(status.last_gate_race_time)


if __name__ == "__main__":
    unittest.main()
