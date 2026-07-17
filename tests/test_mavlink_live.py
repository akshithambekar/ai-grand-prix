import struct
import unittest

from scripts.mavlink_live import TelemetryState


class Message:
    def __init__(self, message_type, **values):
        self._message_type = message_type
        self.__dict__.update(values)

    def get_type(self):
        return self._message_type


class MAVLinkDashboardTests(unittest.TestCase):
    def test_track_packets_are_reassembled_and_exposed(self):
        state = TelemetryState()
        state.update(Message(
            "DATA_TRANSMISSION_HANDSHAKE", width=42, packets=1,
        ))
        gate = struct.pack(
            "<Hfffffffff", 0, 10.0, 1.0, -2.0,
            1.0, 0.0, 0.0, 0.0, 2.5, 3.0,
        )
        payload = bytes((2,)) + struct.pack("<H", 42) + struct.pack("<H", 1) + gate
        state.update(Message(
            "ENCAPSULATED_DATA", data=payload, seqnr=0,
        ))
        self.assertTrue(state.track["valid"])
        self.assertEqual(state.track["gates"][0]["position_ned"], (10.0, 1.0, -2.0))
        self.assertEqual(state.track["gates"][0]["width_m"], 2.5)


if __name__ == "__main__":
    unittest.main()
