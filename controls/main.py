#
# Sample Python client for the AI GP controller
#

import sys
import time
from pathlib import Path

# The project is not packaged, so expose the repository-level scripts package when
# main.py is launched from inside PyAIPilotExample-v2.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reset_hotkey import ResetHotkey
from setup import setup_components

# Modify these properties if you want to run the server remotely for example
SIM_SERVER_UDP_IP = "127.0.0.1"
SIM_SERVER_UDP_PORT = 14550

# time since sim started ms
system_boot_ms = int(time.time() * 1000)

# arbitrary shared data between the various components
shared_data = {}

# setup components
components = setup_components(shared_data, system_boot_ms, SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT)
controller = components['controller']
ts_loop = components['ts_loop']
mavlink_rx = components['mavlink_rx']
vision_rx = components['vision_rx']

print("Arming drone...", flush=True)
controller.arm()
print("Starting control loop...", flush=True)
reset_hotkey = ResetHotkey()
is_running = True
try:
    while is_running:
        if reset_hotkey.consume_request():
            controller.send_sim_reset_command()
            print("Reset command sent.", flush=True)
        controller.update()
except KeyboardInterrupt:
    print("Stopping client...", flush=True)
finally:
    reset_hotkey.close()

# exit
ts_loop.get_thread_for_join().join(timeout=1.0)
mavlink_rx.get_thread_for_join().join(timeout=1.0)
vision_rx.get_thread_for_join().join(timeout=1.0)

print("Client exited!", flush=True)
