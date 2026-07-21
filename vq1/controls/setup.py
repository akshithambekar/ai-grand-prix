from pymavlink import mavutil
from controls.controller import Controller
from controls.mavlink_rx import MAVLinkRX
from controls.timesync import TimeSync
from controls.vision_rx import VisionRX

def setup_components(
    shared_data,
    system_boot_ms,
    server_ip,
    server_udp_port,
    vision_ip="0.0.0.0",
    vision_port=5600,
):
    # -------------------------------
    # Mavlink Connection
    # -------------------------------
    # Start a connection listening on a UDP port
    sim_conn = mavutil.mavlink_connection('udpin:%s:%s' % (server_ip, server_udp_port,))
    print("Waiting for heartbeat...", flush=True)
    sim_conn.wait_heartbeat()
    print(f"Connected to system: {sim_conn.target_system}", flush=True)

    # -------------------------------
    # Setup Mavlink msg receiver
    # -------------------------------
    print("Setting up MAVLink rx...", flush=True)
    mavlink_rx = MAVLinkRX.create_mavlink_rx(sim_conn, shared_data)

    # -------------------------------
    # Timesync request Loop
    # -------------------------------
    print("Setting up Timesync loop...", flush=True)
    ts_loop = TimeSync.create_timesync(sim_conn, shared_data)

    # -------------------------------
    # Connect Vision receiver
    # -------------------------------
    vision_rx = VisionRX(shared_data, bind_ip=vision_ip, bind_port=vision_port)

    # -------------------------------
    # Main control loop
    # -------------------------------
    controller = Controller(sim_conn, shared_data, system_boot_ms)

    return {
        'vision_rx': vision_rx,
        'mavlink_rx': mavlink_rx,
        'ts_loop': ts_loop,
        'sim_conn': sim_conn,
        'controller': controller
    }


def shutdown_components(components):
    for name in ("ts_loop", "mavlink_rx", "vision_rx"):
        component = components.get(name)
        if component is None:
            continue
        thread = component.get_thread_for_join()
        if thread is not None:
            thread.join(timeout=1.0)
    connection = components.get("sim_conn")
    if connection is not None:
        connection.close()
