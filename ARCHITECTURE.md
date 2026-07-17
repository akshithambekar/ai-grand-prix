# Official AI Grand Prix Gymnasium and PPO Architecture

## Goal and runtime constraint

Train a direct-action Stable-Baselines3 PPO policy to pass the gates reliably. Speed
optimization is outside the first objective.

The live system uses one instance of the official Windows-only simulator. The simulator,
Gym environment, and PPO process run on the same Windows machine. Offline unit tests and
CSV reward replay may run on another platform.

```text
Official simulator
  |-- MAVLink UDP 127.0.0.1:14550
  `-- FPV video UDP 0.0.0.0:5600
             |
             v
MAVLinkRX + canonical vehicle/active-gate state + VisionRX/GateTracker
             |
             v
        shared_data
             |
             v
       EpisodeManager
             |
             v
          AIGPEnv
             |
             v
        SB3 direct PPO
             |
             v
bounded SET_ATTITUDE_TARGET commands
```

Only one live client may bind the simulator ports. Telemetry probes, Gym probes, training,
and evaluation therefore run one at a time. SB3 uses one environment and its in-process
`DummyVecEnv` wrapper. `SubprocVecEnv` is excluded.

## Policy interface

### Action

The action space is `Box(-1, 1, shape=(4,), dtype=float32)`:

```text
roll rate, pitch rate, yaw rate, thrust
```

The mapper converts it to conservative absolute commands:

| Command | Range |
|---|---:|
| Roll rate | `[-0.20, 0.20] rad/s` |
| Pitch rate | `[-0.20, 0.15] rad/s` |
| Yaw rate | `[-0.15, 0.15] rad/s` |
| Thrust | `[0.23, 0.30]` |

Zero action produces zero angular rates and `0.265` thrust. PPO runs at 10 Hz. Each Gym
step holds its command for 100 ms and sends it five times at 50 Hz. The controller rejects
non-finite actions, clamps normalized input, and requires explicit command permission.

### Observation

The policy consumes the normalized 38-value `state_v2` vector:

```text
detected
center_x, center_y, log_area, tracking_confidence, missed_frames
vehicle_state_valid
local_position_ned[3], local_velocity_ned[3]
sin/cos(roll, pitch, yaw)
body_rates[3], body_acceleration[3]
track_geometry_valid
active_gate_body_position[3], active_gate_body_normal[3]
gate_width, gate_height
previous_action[4]
```

Centroids are mapped to `[-1, 1]`; area uses normalized log image area. Position,
velocity, rates, acceleration, gate-relative position, and dimensions use fixed scales of
`100 m`, `20 m/s`, `5 rad/s`, `20 m/s^2`, `50 m`, and `10 m`. Missing state groups are
zero only with their validity mask cleared. Odometry in `MAV_FRAME_LOCAL_NED` is preferred;
fresh local position and attitude are fallbacks. NED gate vectors are rotated into body FRD.

## Reward and episodes

Reward components:

```text
-0.01  each 100 ms step
+20.0  each gate-index increment
+50.0  course completion
-20.0  environment or severe gate collision
-2.0   non-terminal gate contact
+/-0.5 maximum true gate-distance progress (visual potential fallback)
up to -0.1 lateral/vertical gate-alignment penalty
```

With valid track geometry, dense reward uses reduction in true active-gate distance and a
bounded gate-opening alignment penalty. If track geometry is absent, it falls back to the
change in this visual potential:

```python
error = sqrt(center_x**2 + center_y**2)
potential = -error
if error < 0.4:
    potential += 0.15 * normalized_log_area
```

It applies only across consecutive detections with the same track and gate. Detection
loss, target changes, gate passage, and reset clear the temporal potential. Resetting
reward state never changes PPO weights.

`EpisodeManager` is the sole reset authority. It handles reset command `31000`, reset
epoch detection, simulator countdown, the four-second command embargo, re-arming, and a
fresh post-reset vision frame. It also terminates on:

- course or configured curriculum completion;
- environment or severe gate collision;
- gate and episode timeout;
- sustained gate loss;
- vision-stream stall.
- sustained inversion or tumbling;
- position divergence, out-of-bounds travel, or a stationary/stuck state.

Physical terminal states return `terminated=True`. Time and data failures return
`truncated=True`. Gym returns the final pre-reset observation, and the next `reset()`
continues the reset handshake already started by `EpisodeManager`.

## Training and evaluation

Initial PPO configuration:

```python
PPO(
    "MlpPolicy",
    env,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,
    policy_kwargs={"net_arch": [128, 128]},
)
```

Curriculum stages require 1, 2, 3, then all gates. Advance after deterministic evaluation
reaches at least 80% success over ten episodes. Evaluation pauses training and reuses the
same simulator. Models, checkpoints, Monitor output, episode metrics, and TensorBoard logs
are written under `artifacts/state_v2/`. Former 19-value policies are intentionally
incompatible and are rejected before evaluation or resume.

## Verification sequence

1. Run all unit tests and SB3 `check_env` against the fake-clock backend.
2. Run `scripts/gym_control_probe.py` on Windows using the proven first-gate command.
3. Confirm one gate reward and a clean target unlock/acquisition after the pass.
4. Run two bounded random episodes with `scripts/random_action_probe.py`.
5. Run a 512-step PPO smoke training session.
6. Train the one-gate curriculum and evaluate ten deterministic episodes.

No flight-control command may be sent before `EpisodeManager.command_allowed` becomes
true. The Windows terminal must remain focused for the `K` emergency reset hotkey.
