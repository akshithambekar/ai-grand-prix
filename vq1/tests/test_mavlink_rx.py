import math
import struct
import unittest
from types import SimpleNamespace

from pymavlink import mavutil

from controls.mavlink_rx import MAVLinkRX


class MAVLinkReceiverTests(unittest.TestCase):
    def setUp(self):
        self.data = {}
        self.rx = MAVLinkRX(None, self.data)

    def test_restored_handlers_publish_snapshots(self):
        self.rx.on_highres_imu(SimpleNamespace(
            time_usec=1, xacc=0.0, yacc=0.0, zacc=9.8,
            xgyro=0.1, ygyro=0.2, zgyro=0.3,
        ))
        self.rx.on_attitude(SimpleNamespace(
            time_boot_ms=1, roll=0.1, pitch=0.2, yaw=0.3,
            rollspeed=0.1, pitchspeed=0.2, yawspeed=0.3,
        ))
        self.rx.on_local_position_ned(SimpleNamespace(
            time_boot_ms=1, x=1.0, y=2.0, z=3.0, vx=4.0, vy=5.0, vz=6.0,
        ))
        self.rx.on_odometry(SimpleNamespace(
            time_usec=1, frame_id=mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            child_frame_id=mavutil.mavlink.MAV_FRAME_BODY_FRD,
            x=1.0, y=2.0, z=3.0, q=(1.0, 0.0, 0.0, 0.0),
            vx=4.0, vy=5.0, vz=6.0, rollspeed=0.1, pitchspeed=0.2, yawspeed=0.3,
            pose_covariance=[0.0] * 21, velocity_covariance=[0.0] * 21,
            reset_counter=4, estimator_type=1, quality=90,
        ))
        self.assertIn("attitude", self.data)
        self.assertIn("local_position_ned", self.data)
        self.assertIn("odometry", self.data)
        self.assertTrue(self.data["vehicle_state"]["valid"])
        self.assertEqual(self.data["vehicle_state"]["position_source"], "odometry")

    def test_track_parser_rejects_invalid_gate(self):
        valid = struct.pack("<Hfffffffff", 0, 10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0)
        invalid = struct.pack("<Hfffffffff", 1, 20.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0)
        self.rx.on_track_data(struct.pack("<H", 2) + valid + invalid)
        self.assertEqual(set(self.data["track"]["gates"]), {0})
        self.assertEqual(self.data["track"]["rejected_gates"], 1)
        self.assertFalse(self.data["track"]["track_geometry_valid"])

    def test_track_parser_rejects_nonfinite_values(self):
        gate = struct.pack("<Hfffffffff", 0, math.nan, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0)
        self.rx.on_track_data(struct.pack("<H", 1) + gate)
        self.assertFalse(self.data["track"]["track_geometry_valid"])

    def test_nonfinite_vehicle_message_does_not_create_valid_state(self):
        self.rx.on_highres_imu(SimpleNamespace(
            time_usec=1, xacc=0.0, yacc=0.0, zacc=9.8,
            xgyro=0.0, ygyro=0.0, zgyro=0.0,
        ))
        self.rx.on_attitude(SimpleNamespace(
            time_boot_ms=1, roll=0.0, pitch=0.0, yaw=0.0,
            rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0,
        ))
        self.rx.on_local_position_ned(SimpleNamespace(
            time_boot_ms=1, x=math.nan, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0,
        ))
        self.assertFalse(self.data["vehicle_state"]["valid"])

    def test_truncated_track_payload_is_rejected(self):
        self.rx.on_track_data(struct.pack("<H", 2) + b"short")
        self.assertEqual(self.data["track"]["rejected_gates"], 2)
        self.assertFalse(self.data["track"]["track_geometry_valid"])


if __name__ == "__main__":
    unittest.main()
