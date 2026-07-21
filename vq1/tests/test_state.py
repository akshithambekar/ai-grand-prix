import math
import unittest

from pymavlink import mavutil

from controls.state import (
    derive_active_gate_state,
    derive_vehicle_state,
    inverse_rotate_vector,
    quaternion_from_euler,
    rotate_vector,
)


def raw_state(now=10.0):
    return {
        "imu": {
            "xacc": 0.0, "yacc": 0.0, "zacc": 9.8, "received_at_s": now - 0.1,
        },
        "attitude": {
            "euler": (0.1, 0.2, 0.3), "body_rates": (1.0, 2.0, 3.0),
            "received_at_s": now - 0.1,
        },
        "local_position_ned": {
            "position_ned": (1.0, 2.0, 3.0), "velocity_ned": (4.0, 5.0, 6.0),
            "received_at_s": now - 0.1,
        },
    }


class VehicleStateTests(unittest.TestCase):
    def test_fresh_compatible_odometry_is_preferred(self):
        data = raw_state()
        data["odometry"] = {
            "frame_id": mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            "child_frame_id": mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            "position_ned": (10.0, 20.0, 30.0), "velocity_ned": (7.0, 8.0, 9.0),
            "quaternion_wxyz": (1.0, 0.0, 0.0, 0.0), "body_rates": (0.1, 0.2, 0.3),
            "reset_counter": 2, "received_at_s": 9.9,
        }
        state = derive_vehicle_state(data, now=10.0)
        self.assertTrue(state["valid"])
        self.assertEqual(state["position_source"], "odometry")
        self.assertEqual(state["position_ned"], (10.0, 20.0, 30.0))
        self.assertEqual(state["odometry_reset_counter"], 2)

    def test_stale_or_incompatible_odometry_falls_back(self):
        data = raw_state()
        data["odometry"] = {
            "frame_id": 99, "position_ned": (10.0, 20.0, 30.0),
            "child_frame_id": mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            "velocity_ned": (7.0, 8.0, 9.0), "quaternion_wxyz": (1.0, 0.0, 0.0, 0.0),
            "body_rates": (0.1, 0.2, 0.3), "received_at_s": 9.9,
        }
        state = derive_vehicle_state(data, now=10.0)
        self.assertTrue(state["valid"])
        self.assertEqual(state["position_source"], "local_position_ned")
        self.assertEqual(state["attitude_source"], "attitude")

    def test_state_expires_instead_of_becoming_zero(self):
        state = derive_vehicle_state(raw_state(now=1.0), now=10.0)
        self.assertFalse(state["valid"])
        self.assertIsNone(state["position_ned"])

    def test_active_gate_transform_and_normal_sign(self):
        data = raw_state()
        data["vehicle_state"] = {
            "valid": True, "position_ned": (0.0, 0.0, 0.0),
            "quaternion_wxyz": quaternion_from_euler(0.0, 0.0, math.pi / 2.0),
        }
        data["race_status"] = {"active_gate_index": 3}
        data["track"] = {
            "track_geometry_valid": True,
            "gates": {3: {
                "gate_id": 3, "position_ned": (0.0, 10.0, 0.0),
                "quaternion_wxyz": (1.0, 0.0, 0.0, 0.0),
                "width_m": 2.0, "height_m": 3.0,
            }},
        }
        gate = derive_active_gate_state(data)
        self.assertTrue(gate["valid"])
        self.assertAlmostEqual(gate["relative_position_body"][0], 10.0, places=5)
        self.assertAlmostEqual(gate["relative_position_body"][1], 0.0, places=5)
        self.assertAlmostEqual(gate["distance_m"], 10.0)
        self.assertAlmostEqual(gate["plane_distance_m"], 0.0)
        self.assertAlmostEqual(gate["lateral_m"], -10.0)

    def test_gate_plane_alignment_is_independent_of_vehicle_yaw(self):
        alignments = []
        for yaw in (0.0, math.pi / 2.0):
            data = {
                "vehicle_state": {
                    "valid": True, "position_ned": (0.0, 1.0, 0.5),
                    "quaternion_wxyz": quaternion_from_euler(0.0, 0.0, yaw),
                },
                "race_status": {"active_gate_index": 0},
                "track": {"track_geometry_valid": True, "gates": {0: {
                    "gate_id": 0, "position_ned": (10.0, 0.0, 0.0),
                    "quaternion_wxyz": (1.0, 0.0, 0.0, 0.0),
                    "width_m": 2.0, "height_m": 2.0,
                }}},
            }
            gate = derive_active_gate_state(data)
            alignments.append((
                gate["plane_distance_m"], gate["lateral_m"], gate["vertical_m"],
            ))
        self.assertEqual(alignments[0], alignments[1])

    def test_inverse_rotation_round_trip(self):
        q = quaternion_from_euler(0.3, -0.2, 1.1)
        original = (1.0, 2.0, 3.0)
        body = inverse_rotate_vector(q, original)
        restored = rotate_vector(q, body)
        for actual, expected in zip(restored, original):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_roll_transform_maps_parent_down_to_body_right(self):
        q = quaternion_from_euler(math.pi / 2.0, 0.0, 0.0)
        body = inverse_rotate_vector(q, (0.0, 0.0, 10.0))
        self.assertAlmostEqual(body[0], 0.0, places=5)
        self.assertAlmostEqual(body[1], 10.0, places=5)
        self.assertAlmostEqual(body[2], 0.0, places=5)

    def test_gate_normal_is_consistent_for_flipped_quaternions(self):
        normals = []
        for gate_q in (
            (1.0, 0.0, 0.0, 0.0),
            quaternion_from_euler(0.0, 0.0, math.pi),
        ):
            data = {
                "vehicle_state": {
                    "valid": True, "position_ned": (0.0, 0.0, 0.0),
                    "quaternion_wxyz": (1.0, 0.0, 0.0, 0.0),
                },
                "race_status": {"active_gate_index": 0},
                "track": {
                    "track_geometry_valid": True,
                    "gates": {0: {
                        "gate_id": 0, "position_ned": (10.0, 0.0, 0.0),
                        "quaternion_wxyz": gate_q, "width_m": 2.0, "height_m": 2.0,
                    }},
                },
            }
            normals.append(derive_active_gate_state(data)["gate_normal_body"])
        for normal in normals:
            self.assertAlmostEqual(normal[0], -1.0, places=5)
            self.assertAlmostEqual(normal[1], 0.0, places=5)

    def test_missing_active_gate_id_marks_geometry_invalid(self):
        data = {
            "vehicle_state": {
                "valid": True, "position_ned": (0.0, 0.0, 0.0),
                "quaternion_wxyz": (1.0, 0.0, 0.0, 0.0),
            },
            "race_status": {"active_gate_index": 7},
            "track": {"track_geometry_valid": True, "gates": {}},
        }
        self.assertFalse(derive_active_gate_state(data)["valid"])


if __name__ == "__main__":
    unittest.main()
