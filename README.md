# AI-GP Round 1 

VQ1 simulator integration by **Abhinav Maddi** (GitHub:
[@abhimaddi](https://github.com/abhimaddi)).

[![demo](./demo.gif)]

| config | gates | time | collisions |
|---|---:|---:|---:|
| `measured` | `6/6` | `24.xx s` | `0` |

## Run

`fly.py` listens for simulator telemetry, arms the drone, waits for a fresh race
start, then sends rate/thrust commands through MAVLink.

### Windows + VQ1 simulator

Put the extracted VQ1 simulator beside this repository (the default expected
path is `../AIGP_VQ1_3385/FlightSim.exe`), then run this from PowerShell:

```powershell
.\run-vq1.ps1
```

The launcher creates a native Windows virtual environment, installs the pilot,
and opens the simulator. In VQ1, log in, enter Virtual Qualifier Round 1, reset
the drone to the start line, and start the countdown. Leave the PowerShell
window running; the pilot waits until VQ1 begins publishing race telemetry.

If the simulator is already open, or is installed elsewhere:

```powershell
.\run-vq1.ps1 -SimulatorPath "C:\path\to\VQ1\FlightSim.exe"
.\run-vq1.ps1 -SkipSimulator
.\run-vq1.ps1 -SkipSimulator -CheckOnly
```

The pilot listens on UDP `14550`, matching the official `PyAIPilotExample`.
Seeing VQ1 itself bound to source port `14560` is normal.

If PowerShell blocks local scripts, run this once in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Python from python.org (3.11 or newer) is recommended. MSYS/Git Bash Python is
not compatible with the Windows NumPy wheels used by this project.

### Linux or a separate pilot machine

The simulator runs on Windows. The pilot can run on another machine if a UDP
relay forwards the simulator's localhost MAVLink stream:

```cmd
python relay.py --target-ip <PILOT_IP>
```

Then start the pilot:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
python3 fly.py
```

Start the simulator run after the pilot is waiting.

## Files

- `fly.py`
- `run-vq1.ps1`
- `relay.py`
- `src/aigp_pilot/raceconfig.py`
- `src/aigp_pilot/course.py`
- `src/aigp_pilot/raceline.py`
- `src/aigp_pilot/control.py`
- `src/aigp_pilot/parsers.py`

## Test

```bash
python3 -m pytest -q
```
