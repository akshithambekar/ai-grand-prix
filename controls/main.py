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
from episode_manager import EpisodeManager
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
episodes = EpisodeManager(
    shared_data,
    send_reset=controller.send_sim_reset_command,
    send_arm=controller.arm,
)

print("Preparing initial episode...", flush=True)
reset_hotkey = ResetHotkey()
episodes.request_reset(reason="initial")
print("Starting episode loop...", flush=True)
is_running = True
try:
    while is_running:
        if reset_hotkey.consume_request():
            episodes.request_reset(reason="manual")
            print("Reset requested.", flush=True)
        event = episodes.update()
        if event.episode_started:
            print(f"Episode {event.episode_id} started.", flush=True)
        if event.episode_ended:
            print(f"Episode ended: {event.reason}", flush=True)
        if event.command_allowed:
            controller.update()
        else:
            time.sleep(1.0 / 250.0)
except KeyboardInterrupt:
    print("Stopping client...", flush=True)
finally:
    reset_hotkey.close()

# exit
ts_loop.get_thread_for_join().join(timeout=1.0)
mavlink_rx.get_thread_for_join().join(timeout=1.0)
vision_rx.get_thread_for_join().join(timeout=1.0)

print("Client exited!", flush=True)
