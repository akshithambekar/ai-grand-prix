# Training resets and measurements for the AI drone

Automated episode-termination conditions for training, plus what to measure per episode.

## Prerequisite: telemetry is not reaching `shared_data`

Most of the triggers below are not implementable yet.
Every handler in `mavlink_rx.py` unpacks its message into local variables and then drops them on the floor.
`on_race_status` (mavlink_rx.py:205) parses `active_gate_index` and `last_gate_race_time` and discards both.
`on_collision` (mavlink_rx.py:263) parses `collision_id`, `threat_level` and the impulse magnitude and discards all three.
Vision is the only component actually publishing, via `self.data["gate"]`.

Step one is plumbing race status and collisions into `shared_data`, following the atomic dict swap convention `vision_rx` already uses.
After that, most of the triggers below are a few lines each.

## Constraint: what is actually observable

ATTITUDE, LOCAL_POSITION_NED and ODOMETRY are all disabled in the current sim config.
There is no position, no velocity, and no attitude on the wire.

Everything below has to be built from HIGHRES_IMU, COLLISION, race status, and vision.
That rules out the obvious triggers like "altitude below floor" or "strayed more than X meters off track".
Only what is observable is listed here.

## Reset triggers, roughly in order of value

### Per-gate time budget

Build this first, ahead of everything else.
If `active_gate_index` has not advanced within N seconds of the last gate pass, reset.
It catches every failure mode that does not crash the drone: orbiting, stalling, wandering off, hovering.
It does it with one integer comparison.

Start N generous, maybe 15s, and ratchet it down as the policy improves.
A tightening budget is itself a curriculum.

### Environment collision

`collision_id == 1002` is an unambiguous crash.
Reset immediately.
Log `threat_level` and the impulse so a scrape can be told apart from a wall strike.

### Gate collision

`collision_id == 1001` is more interesting, and resetting on it is worth resisting at first.
Clipping a gate rail is a near miss, not a failure, and it is exactly the behavior the drone should learn its way off of gradually.
Reset only on `threat_level == 2`, and treat level 1 as a logged penalty.
Revisit once it is known whether the race rules actually disqualify a gate touch.

### Tumbling

Gyro magnitude from HIGHRES_IMU above a threshold sustained over a window means unrecoverable spin.
The drone will never see a gate again, so every frame after that point is wasted rollout.

### Inverted

When accel magnitude is near 1g the drone is quasi-static and the reading is mostly gravity, so the gravity vector gives tilt directly.
Sustained tilt past roughly 90 degrees is a reset.
This is unreliable during aggressive maneuvers, which is why it needs the quasi-static gate.
It catches the slow flip-and-drift that tumbling detection misses.

### Range divergence

`range_m` is published per frame.
If it increases monotonically over a rolling window while a gate is detected, the drone is flying away from its target.

This complements the "no gates visible" trigger, since it fires while the gate is still in frame.
That is much earlier, and with cleaner credit assignment.

### Centroid pinned to frame edge

If the gate centroid sits within a few percent of the frame border for K consecutive frames, the gate is on its way out of the FOV.
Fires one beat before "no gates visible" does.

### Stuck

Accel reading close to pure gravity plus `vision_velocity` near zero for several seconds means the drone is wedged or hovering.
Distinct from the time budget in that it fires faster.

### Vision stream stall

No decoded frame for X ms.
This is not a policy failure, it is a plumbing failure.
It needs its own reset reason so it does not pollute the training statistics.

### Wall-clock episode cap

Always have one, as a backstop for whatever was not anticipated.

## Refinement to the existing "no gates visible" trigger

As written it will fire spuriously in three legitimate situations:

- At race start, before the drone is oriented.
- During the moment of passing through a gate, when the gate fills and then leaves the frame.
- Mid hard turn.

So it will punish the drone for succeeding.

Gate it behind a grace period that is suppressed for roughly 0.5s after `active_gate_index` advances, and require K consecutive no-detect frames rather than one.
The README notes the detector holds a gate down to about 26px of side, so genuine loss of sight is a real signal.
It is just worth not confusing it with the two frames after a clean pass.

## What to measure

The single most valuable metric is a **histogram of reset reasons per session**.
If 80% of resets are "no gates visible", the other triggers are dead code and training is happening against one failure mode without anyone knowing.
It also reveals a mistuned threshold, because a trigger that fires either constantly or never is a trigger with a bad constant.

Per episode:

- Gates passed.
- Per-gate split times.
- Total race time.
- Detection rate: frames with a gate over total frames.
- Collisions, bucketed by type and threat level.
- Peak and mean gyro magnitude.
- Centroid offset at the moment of gate pass, as a proxy for how centered the drone flew through.

On the health side, log the achieved control loop rate against the nominal 250Hz and the achieved vision FPS against 30.
If either sags under load, the training data is quietly wrong about the timing.

## Verify before trusting any of this

`send_sim_reset_command` exists at controller.py:160 but nothing appears to call it.
Confirm empirically what it actually resets: does `active_gate_index` return to 0, does `race_start_boot_time_ms` clear, and does the drone need re-arming afterward?
`main.py` arms once before the loop and never again.

There also needs to be a way to detect that a reset has *completed* before recording the next episode.
Otherwise transitions that straddle the boundary get logged, which are physically impossible and will poison the buffer.
Watching for `active_gate_index` to return to 0 is the obvious candidate, assuming the reset does that.

## Open work

`main.py` has no episode concept at all.
It is a bare `while is_running` loop that never terminates.
An episode runner has to sit around `controller.update()` before any of this can be wired up.
