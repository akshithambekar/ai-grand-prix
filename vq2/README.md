# AI Grand Prix Virtual Qualifier 2

## Environment

From this directory on the Windows simulator machine:

```powershell
uv sync
```

Only one program may bind MAVLink UDP port `14550` and FPV UDP port `5600` at a time.
Stop the starter client, another probe, or another capture process before running these tools.

## 1. Passive simulator contract capture

This sends no reset, arm, or flight-control message:

```powershell
uv run python scripts\simulator_contract_probe.py
```

It listens for eight seconds, saves every complete FPV JPEG, records every MAVLink
message, reconstructs track packets to verify whether gate geometry is nulled, reports
whether the cyan guide appears, and writes `summary.json` under
`artifacts\contract\contract_<timestamp>`.

## 2. Bounded active contract probe

Test the attitude-rate interface first:

```powershell
uv run python scripts\simulator_contract_probe.py --execute --mode attitude
```

The active probe resets the simulator, waits through the countdown, arms, sends a short
forward-pitch pulse between two neutral phases, then resets. The summary contains IMU
ranges for each phase. If the attitude command produces no response, test body-velocity
control:

```powershell
uv run python scripts\simulator_contract_probe.py --execute --mode velocity
```

Raw-motor mode requires an explicitly measured hover value because an unsafe default would
invalidate the test and may immediately crash:

```powershell
uv run python scripts\simulator_contract_probe.py --execute --mode motor --motor-hover 0.50
```

## 3. Manual synchronized capture

After selecting the working command interface:

```powershell
uv run python scripts\manual_capture.py --execute --mode attitude
```

Controls:

| Key | Motion |
|---|---|
| `W` | Forward |
| `A` | Left |
| `S` | Back |
| `D` | Right |
| `Q` | Up |
| `E` | Down |
| `R` | Reset and wait through the command embargo |
| `Escape` | Finish capture |

Releasing every movement key returns to the configured neutral command. The tool resets on
startup and exit by default. Pass `--no-reset-before-run` or `--no-reset-on-exit` only when
preserving simulator state is intentional.

Each run under `artifacts\manual\manual_<timestamp>` contains:

- `frames\<frame_id>_<sim_time_ns>.jpg`: every complete FPV frame;
- `events.jsonl`: one synchronized timeline of MAVLink, frames, keys, commands, resets,
  and race-status records;
- `summary.json`: stream counts, frame counts, cyan-guide result, and command totals.

Every event has `host_time_ns` and `host_monotonic_ns`. Frame events also retain simulator
`sim_time_ns`; MAVLink events retain all original fields and their simulator timestamps.
Ten `TIMESYNC` requests per second are recorded in the same timeline so the server/client
clock relationship can be reconstructed rather than inferred from UDP arrival order alone.
