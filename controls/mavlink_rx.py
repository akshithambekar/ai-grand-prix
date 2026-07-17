import struct
import time
import threading
import math

from pymavlink import mavutil

from controls.state import derive_vehicle_state

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID  = 2

class MAVLinkRX:

    def __init__(self, mavlink_connection, data):
        self.mavlink_conn = mavlink_connection
        self.data = data
        self.thread = None
        self.is_running = False

        self.track_chunks = {}
        self.expected_num_track_chunks = {}
        self.collision_sequence = 0

    @classmethod
    def create_mavlink_rx(cls, mavlink_connection, data):
        rx = cls(mavlink_connection, data)
        rx.thread = threading.Thread(
            target=rx.mavlink_receive_loop,
            daemon = False
        )
        rx.is_running = True
        rx.thread.start()
        return rx

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def mavlink_receive_loop(self):
        """
        Continuously receive MAVLink messages without blocking.
        """
        while self.is_running:

            try:
                msg = self.mavlink_conn.recv_match(blocking=False)
            except ConnectionResetError:
                print('WARNING: ConnectionResetError was thrown. No longer listening to MAVLink port.')
                return

            if msg is None:
                time.sleep(0.001)
                continue

            msg_type = msg.get_type()

            if msg_type == "BAD_DATA":
                continue

            # --------------------------------------------------------------------------------------
            # HEARTBEAT
            # --------------------------------------------------------------------------------------
            if msg_type == "HEARTBEAT":
                self.on_heartbeat(msg)

            # --------------------------------------------------------------------------------------
            # TIMESYNC
            # --------------------------------------------------------------------------------------
            elif msg_type == "TIMESYNC":
                self.on_timesync(msg)

            # --------------------------------------------------------------------------------------
            # ATTITUDE
            # --------------------------------------------------------------------------------------
            elif msg_type == "ATTITUDE":
                self.on_attitude(msg)

            # --------------------------------------------------------------------------------------
            # LOCAL_POSITION_NED
            # --------------------------------------------------------------------------------------
            elif msg_type == "LOCAL_POSITION_NED":
                self.on_local_position_ned(msg)

            # --------------------------------------------------------------------------------------
            # ODOMETRY
            # --------------------------------------------------------------------------------------
            elif msg_type == "ODOMETRY":
                self.on_odometry(msg)

            # --------------------------------------------------------------------------------------
            # HIGHRES_IMU
            # --------------------------------------------------------------------------------------
            elif msg_type == "HIGHRES_IMU":
                self.on_highres_imu(msg)

            # --------------------------------------------------------------------------------------
            # ENCAPSULATED_DATA
            # --------------------------------------------------------------------------------------
            elif msg_type == "ENCAPSULATED_DATA":
                self.on_encapsulated_data(msg)

            # --------------------------------------------------------------------------------------
            # ACTUATOR_OUTPUT_STATUS
            # --------------------------------------------------------------------------------------
            elif msg_type == "ACTUATOR_OUTPUT_STATUS":
                self.on_actuator_output_status(msg)

            # --------------------------------------------------------------------------------------
            # COLLISION
            # --------------------------------------------------------------------------------------
            elif msg_type == "COLLISION":
                self.on_collision(msg)

            # --------------------------------------------------------------------------------------
            # DATA_TRANSMISSION_HANDSHAKE - Repurposed and used for upcoming 'Track Data' packets
            # --------------------------------------------------------------------------------------
            elif msg.get_type() == "DATA_TRANSMISSION_HANDSHAKE":
                track_data_transfer_id = msg.width
                self.track_chunks[track_data_transfer_id] = {}
                self.expected_num_track_chunks[track_data_transfer_id] = msg.packets

    def on_heartbeat(self, msg):
        armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        self.data["armed"] = armed
        self.data["heartbeat"] = {
            "armed": armed,
            "received_at_s": time.monotonic(),
        }

    def on_timesync(self, msg):
        request_time = msg.ts1
        response_time = msg.tc1

    def on_attitude(self, msg):
        self.data["attitude"] = {
            "time_boot_ms": msg.time_boot_ms,
            "euler": (msg.roll, msg.pitch, msg.yaw),
            "body_rates": (msg.rollspeed, msg.pitchspeed, msg.yawspeed),
            "received_at_s": time.monotonic(),
        }
        derive_vehicle_state(self.data)

    def on_local_position_ned(self, msg):
        self.data["local_position_ned"] = {
            "time_boot_ms": msg.time_boot_ms,
            "position_ned": (msg.x, msg.y, msg.z),
            "velocity_ned": (msg.vx, msg.vy, msg.vz),
            "received_at_s": time.monotonic(),
        }
        derive_vehicle_state(self.data)

    def on_odometry(self, msg):
        self.data["odometry"] = {
            "time_usec": msg.time_usec,
            "frame_id": msg.frame_id,
            "child_frame_id": msg.child_frame_id,
            "position_ned": (msg.x, msg.y, msg.z),
            "quaternion_wxyz": tuple(msg.q),
            "velocity_ned": (msg.vx, msg.vy, msg.vz),
            "velocity_frame_id": msg.child_frame_id,
            "body_rates": (msg.rollspeed, msg.pitchspeed, msg.yawspeed),
            "pose_covariance": tuple(getattr(msg, "pose_covariance", ())),
            "velocity_covariance": tuple(getattr(msg, "velocity_covariance", ())),
            "reset_counter": msg.reset_counter,
            "estimator_type": getattr(msg, "estimator_type", None),
            "quality": getattr(msg, "quality", None),
            "received_at_s": time.monotonic(),
        }
        derive_vehicle_state(self.data)

    def on_highres_imu(self, msg):
        self.data["imu"] = {
            "time_usec": msg.time_usec,
            "xacc": msg.xacc,
            "yacc": msg.yacc,
            "zacc": msg.zacc,
            "xgyro": msg.xgyro,
            "ygyro": msg.ygyro,
            "zgyro": msg.zgyro,
            "received_at_s": time.monotonic(),
        }
        derive_vehicle_state(self.data)

    def on_encapsulated_data(self, msg):
        if msg:
            raw_payload = bytes(msg.data)
            data_type = raw_payload[0]

            if int(data_type) == ENCAPSULATED_RACE_STATUS_MSG_ID:
                self.on_race_status(msg)
            elif int(data_type) == ENCAPSULATED_TRACK_INFO_MSG_ID:
                self.on_track_data_packet(msg)

    def on_race_status(self, msg):
        raw_payload = bytes(msg.data)
        # data_type - ID of this message
        # sim_boot_time_ms - elapsed ms on server since sim boot
        # race_start_boot_time_ms - elapsed ms on server since sim boot when race started. None or < 0 if race has not started
        # race_finish_time_ns - elapsed ns on server since sim boot when race finished. None or < 0 if race is ongoing
        # active_gate_index - current index of target race gate
        # last_gate_race_time - race time in seconds when last gate was passed
        data_type, sim_boot_time_ms, race_start_boot_time_ms, race_finish_time_ns, active_gate_index, last_gate_race_time = struct.unpack_from(
            "<BQqqIq", raw_payload)
        self.data["race_status"] = {
            "sim_boot_time_ms": sim_boot_time_ms,
            "race_start_boot_time_ms": race_start_boot_time_ms,
            "race_finish_time_ns": race_finish_time_ns,
            "active_gate_index": active_gate_index,
            "last_gate_race_time": last_gate_race_time,
            "received_at_s": time.monotonic(),
        }
        derive_vehicle_state(self.data)

    def on_track_data_packet(self, msg):
        raw_payload = bytes(msg.data)
        # header:
        #   data_type - ID of this message
        #   transfer_id - ID of the group of packets this chunk belongs to
        data_type, transfer_id = struct.unpack_from("<BH", raw_payload)
        if transfer_id not in self.expected_num_track_chunks:
            return
        raw_payload = raw_payload[3:]
        if msg.seqnr < 0 or msg.seqnr >= self.expected_num_track_chunks[transfer_id]:
            return
        self.track_chunks[transfer_id][msg.seqnr] = raw_payload
        if len(self.track_chunks[transfer_id]) == self.expected_num_track_chunks[transfer_id]:
            full_payload = b"".join(
                self.track_chunks[transfer_id][i]
                for i in range(self.expected_num_track_chunks[transfer_id])
            )
            del self.track_chunks[transfer_id]
            del self.expected_num_track_chunks[transfer_id]
            self.on_track_data(full_payload)

    def on_track_data(self, payload):
        # header:
        #   num_gates - track gate count
        record_format = "<Hfffffffff"
        record_size = struct.calcsize(record_format)
        if len(payload) < 2:
            return
        num_gates, = struct.unpack_from("<H", payload)
        payload = payload[2:]
        gates = {}
        rejected = 0
        for i in range(num_gates):
            # Gate Info
            #   gate_id - range is 0 - num_gates
            #   position_ned_x, position_ned_y, position_ned_z - Position of gate in NED coordinates
            #   orientation_ned_w, orientation_ned_x, orientation_ned_y, orientation_ned_z - Orientation of gate in NED coordinates
            #   width - gate width in metres
            #   height - gate height in metres
            if len(payload) < record_size:
                rejected += num_gates - i
                break
            values = struct.unpack_from(record_format, payload)
            payload = payload[record_size:]
            gate_id, position_ned_x, position_ned_y, position_ned_z, orientation_ned_w, orientation_ned_x, orientation_ned_y, orientation_ned_z, width, height = values
            numeric = values[1:]
            q_norm = math.sqrt(sum(value * value for value in values[4:8]))
            if not all(math.isfinite(value) for value in numeric) or width <= 0 or height <= 0 or q_norm < 1e-6:
                rejected += 1
                continue
            gates[int(gate_id)] = {
                "gate_id": int(gate_id),
                "position_ned": (position_ned_x, position_ned_y, position_ned_z),
                "quaternion_wxyz": (
                    orientation_ned_w / q_norm,
                    orientation_ned_x / q_norm,
                    orientation_ned_y / q_norm,
                    orientation_ned_z / q_norm,
                ),
                "width_m": width,
                "height_m": height,
            }
        self.data["track"] = {
            "num_gates": int(num_gates),
            "gates": gates,
            "rejected_gates": rejected,
            "track_geometry_valid": bool(num_gates and len(gates) == num_gates and rejected == 0),
            "received_at_s": time.monotonic(),
        }
        derive_vehicle_state(self.data)

    def on_actuator_output_status(self, msg):
        self.data["actuator_output"] = {
            "time_usec": msg.time_usec,
            "motors": tuple(msg.actuator[:4]),
            "active": getattr(msg, "active", None),
            "received_at_s": time.monotonic(),
        }

    def on_collision(self, msg):
        # Collision IDs
        # 1001 - Gate
        # 1002 - Environment
        self.collision_sequence += 1
        self.data["collision"] = {
            "sequence": self.collision_sequence,
            "collision_id": msg.id,
            "threat_level": msg.threat_level,
            # MAVLink reuses this field for collision impulse magnitude in kg m/s.
            "impulse": msg.horizontal_minimum_delta,
            "received_at_s": time.monotonic(),
        }
