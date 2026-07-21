#
# Sample Python client for the AI GP controller
#

import sys
import time
from pathlib import Path

# The project is script-oriented, so expose the repository root for direct execution.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reset_hotkey import ResetHotkey
from controls.controller import FlightCommand
from controls.episode_manager import EpisodeManager
from controls.setup import setup_components, shutdown_components

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
hover_command = FlightCommand(0.0, 0.0, 0.0, 0.265)
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
            controller.send_flight_command(hover_command, event.command_allowed)
        time.sleep(1.0 / 50.0)
except KeyboardInterrupt:
    print("Stopping client...", flush=True)
finally:
    reset_hotkey.close()
    if episodes.command_allowed:
        episodes.request_reset(reason="client_close")
    shutdown_components(components)

print("Client exited!", flush=True)
