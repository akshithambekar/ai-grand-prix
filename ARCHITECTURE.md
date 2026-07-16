# AI Grand Prix PPO Architecture

## Objective

Fly the fixed AI Grand Prix course reliably and pass every gate. Lap-time optimization is out of scope. The policy must not depend on disabled odometry, local-position, attitude, or gate-coordinate telemetry.

## System overview

```text
FPV camera UDP
      |
      v
HSV red segmentation -> gate candidates -> target tracker
                                         |
HIGHRES_IMU -----------------------------+--> normalized observation
previous action -------------------------+            |
                                                      v
visual-servo baseline + PPO residual -> safety limits -> MAVLink SET_ATTITUDE_TARGET

race status -----------------------------> reward and episode state only
collision messages ----------------------> penalty and termination
reset hotkey / episode manager ----------> MAV_CMD 31000
```

The existing vision pipeline is sufficiently clean that the first policy should consume extracted gate geometry rather than a binary image. This keeps the model and simulator sample requirement small. The binary mask remains useful for debugging, offline replay, and future CNN experiments.

## Perception and target tracking

`vision_rx.py` already produces a red mask, filters gate-like contours, selects the largest candidate, estimates geometry, and publishes an immutable snapshot in `shared_data["gate"]`.

Add a small target tracker before control:

1. Retain every valid gate candidate, sorted by contour area.
2. Associate the previous target using centroid distance, size ratio, and bounding-box overlap.
3. Prefer the associated candidate over the largest candidate while the association remains plausible.
4. Reject clipped edge fragments during gate passage.
5. After a large centered target exits the frame, enter a 3-5 frame cooldown and acquire the next centered candidate.
6. Clear tracking state when `active_gate_index` advances.

Use the inner-hole center or `minAreaRect` center when available. A contour centroid can shift when one gate rail is clipped.

## Policy observation

Use a normalized vector:

```text
detected
center_x, center_y
log_width, log_height, log_area
aspect_ratio, has_hole
delta_center_x, delta_center_y, delta_log_area
frames_since_detection
gyro_x, gyro_y, gyro_z
accel_x, accel_y, accel_z
previous_action[4]
```

Normalize image geometry as:

```text
center_x = 2 * centroid_x / image_width - 1
center_y = 2 * centroid_y / image_height - 1
width    = bbox_width / image_width
height   = bbox_height / image_height
area     = contour_area / (image_width * image_height)
```

Initially exclude `range_m`, `gate_body_pos`, `vision_velocity`, and `pnp_rvec` from the policy. Gate width and camera field of view are currently assumptions. Their errors directly scale metric estimates. The current vision-velocity estimate also mixes camera rotation with vehicle translation and can spike during a target switch.

## Baseline controller

Implement a conservative visual servo:

- Horizontal gate error controls yaw rate and a small roll correction.
- Vertical gate error controls thrust around calibrated hover thrust.
- A small forward-pitch command advances only while the gate is reasonably centered.
- Forward pitch decreases as apparent gate area grows.
- Sustained gate loss stops forward motion and begins a slow search.
- Attitude rates and thrust remain inside conservative safety bounds.

This controller should pass at least the first gate consistently before PPO training. It provides safe exploration and carries the vehicle far enough into the course for later gates to enter the training distribution.

## PPO policy

Use Stable-Baselines3 `PPO` with `MlpPolicy`. The action is a four-dimensional residual in `[-1, 1]`:

```text
delta_roll_rate, delta_pitch_rate, delta_yaw_rate, delta_thrust
```

Initial residual authority:

```text
roll rate:  +/- 0.15 rad/s
pitch rate: +/- 0.15 rad/s
yaw rate:   +/- 0.12 rad/s
thrust:     +/- 0.03
```

Run policy inference at 10 Hz. Hold the combined command and publish it through `SET_ATTITUDE_TARGET` at 50 Hz. Apply action smoothing, rate limits, and final clamps after adding the residual.

Suggested starting configuration:

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

## Reward

Let `e` be normalized gate-center error and `s` be log normalized area:

```text
e = sqrt(center_x^2 + center_y^2)
s = log(area + epsilon)

potential = -e + 0.15 * s       when e < 0.4
potential = -e                  otherwise

reward = gamma * potential_next - potential_current
         - 0.01
         + 20.0 * gate_index_increment
         + 50.0 * course_complete
         - 20.0 * collision
```

Clip per-step area progress so a target switch cannot create a reward spike. Race status is used for reward and episode bookkeeping, never as a policy observation. Do not add a survival reward because it encourages hovering.

## Limited-simulator training

Use one `DummyVecEnv` and a 100 ms action step. Train with feature-level randomization:

- centroid noise;
- 5-10% size and area noise;
- one-to-three-frame detection dropouts;
- observation delay;
- one-frame target-switch outliers;
- small IMU and action noise;
- hover-thrust bias.

Training progression:

1. Validate segmentation and target switching with offline replay.
2. Make the visual servo pass gate one consistently.
3. Run the full course with PPO residual scale set to zero.
4. Begin PPO with small residual authority.
5. Increase authority only while deterministic evaluation remains stable.
6. Evaluate every tenth episode with feature noise disabled.

An episode ends on course completion, collision, unrecoverable stall, or sustained gate loss. Reset with MAVLink command `31000`, then wait for fresh race status and re-arm. If the qualifier ignores the reset command, automate the simulator restart UI and retain MAVLink for flight control.

