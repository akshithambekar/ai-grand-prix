# AI Grand Prix — Virtual Qualifier 1: System Plan

A working summary of the autonomy stack design for the AI Grand Prix drone race.

## Context

- **Goal:** autonomous software to fly a drone through gates as fast as possible.
- **Round One bar:** just *complete* the 3-gate course within 8 minutes. Speed matters for ranking later, not for qualifying.
- **Practice sim:** the open-source `elodin-sys/ai-grand-prix` rig (Betaflight SITL + Elodin physics), which mirrors the official Anduril sim.
- **The contract:** you implement one function, `autopilot(SensorUpdate) -> RCCommand`, called every physics tick (120 Hz).
  - `SensorUpdate` gives IMU, velocity, attitude, baro, mag, an optional camera frame (30 Hz), and — in the practice rig only — a ground-truth world pose.
  - `RCCommand` is four RC stick values (throttle / roll / pitch / yaw, 1000–2000) plus arm. These feed Betaflight, which closes the inner stabilization loop.

## Key early decisions

- **The output is RC sticks, not motor commands.** Betaflight already handles low-level stabilization, so the control layer is thin.
- **No absolute position at race time.** The official sim exposes velocity and attitude but not position. The practice rig's world pose is an **oracle for training labels only** — never an input the deployed system relies on, or it won't transfer.
- **Strategy: build a learned (RL) system now, not a minimal one.** A learned stack has high fixed cost but scales to harder tracks and later rounds by retraining rather than rewriting.
- **"End-to-end" means end-to-end *control*, not pixels-to-motors.** The policy learns to fly from an abstract state; perception stays a separate, supervised module. (This is the UZH "Swift" recipe that beat human champions.)
- **Hedge:** keep a simple classical "fly toward the detected gate" controller as a qualifying fallback. It shares the same perception front end, so it costs little.

## The system: four components

```
camera image ──▶ [1] Gate detector ──▶ [2] State estimator ──▶ [3] Flight policy ──▶ sticks
                                            ▲
                              IMU / velocity ┘
                                                          all wired together by [4] the glue
```

**1. Gate detector** — *model: small YOLO-style / keypoint CNN.*
Finds the next gate's four corners in the image. Trained on sim images auto-labeled from the ground-truth gate positions. Corners + the known 1.5 m gate size → a standard PnP solve → the gate's pose relative to you. (CNN is the model; PnP is just geometry.)

**2. State estimator** — *not a learned model: an Extended Kalman Filter (EKF).*
Fuses IMU + velocity with the detector's gate sightings to answer "where am I and how am I moving." Runs every tick; corrects itself whenever a fresh gate sighting arrives.

**3. Flight policy** — *model: small MLP trained with PPO (reinforcement learning).*
Takes the state estimate + relative next-gate pose, decides how to fly. Outputs thrust + body-rotation-rate commands, which are then mapped to the four stick values.

**4. The glue** — *the `autopilot()` function.*
Each tick: run the camera through the detector (only on fresh frames), update the estimator, feed the result to the policy, return the sticks. Plus the trivial conversions to/from the sim's data types.

## Build order

1. **Flight policy first.** It's the long pole and needs no camera. Train it in a fast, simple drone physics sim you write yourself (not the competition sim), feeding it the *true* state — millions of fast practice flights.
2. **Gate detector.** Trained on labeled sim images. Replaces the "true state" training cheat with something usable for real.
3. **State estimator.** Stitches the detector's sightings together with the sensors so the policy receives a clean state.
4. **The glue.** Wire all three into `autopilot()` and run it in the actual competition sim.

**Why this order:** the policy is the hardest and most independent, so start it early. The detector and estimator exist to feed the policy a believable state once the oracle-pose cheat is removed. The glue is quick once the pieces exist.

## Constraints worth remembering

- Camera is tilted **20° upward**; field of view is ~90° horizontal / ~59° vertical (the spec's "VFoV 90°" is mislabeled — trust the intrinsics).
- Camera runs at **30 Hz**, control at **120 Hz** — run heavy perception only on fresh frames; propagate the estimator every tick.
- In `RCCommand`, **pitch < 1500 = forward**, and the drone must be **armed** before it flies.
- Practice rig is **ENU**; official spec is **NED** — centralize all frame conversions in one place.
- The biggest risk is the **sim-to-sim gap** between your fast training sim and the real one. Close it by modeling latency, randomizing dynamics, and injecting your perception's real error profile into training.
