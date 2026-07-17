Code for Anduril's AI Grand Prix

## Setup

Install [uv](https://docs.astral.sh/uv/) and sync the locked environment:

```bash
uv sync
```

Run the MAVLink client with:

```bash
uv run python PyAIPilotExample-v2/main.py
```

Run the standalone reset listener with:

```bash
uv run python PyAIPilotExample-v2/reset_hotkey.py
```

On Windows, keep the terminal window focused while pressing `K`.

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

# Progress tracker

## Vision: red gate detection

Component: `PyAIPilotExample-v2/vision_rx.py`

Status: implemented and working on a real frame, but uncalibrated.
Bearing is trustworthy now.
Absolute range is not, until the two constants below are confirmed.

### Done

- [x] HSV threshold on the gate's red hue, two bands OR'd together because red wraps the hue circle.
- [x] Morphological close then open to knit the hollow frame's rails together and kill speckle.
- [x] Contour filtering on area, aspect ratio and hollowness, picking the largest survivor as the nearest gate.
- [x] Published to `self.data["gate"]`: bbox, centroid, `range_m`, `gate_body_pos`, `pnp_ok`, `pnp_rvec`, `vision_velocity`.
- [x] Explicit no-detection state, so the controller can tell a lost gate from a stale one.
- [x] Thresholds tuned against a real frame rather than assumed.
- [x] Confirmed the detector does not latch onto the cyan racing line, the blue panel, or the yellow HUD.

### Open

- [ ] **Calibrate `GATE_OUTER_WIDTH_M`** (currently an assumed 1.5m).
      Gate dimensions are nulled in telemetry as of the current simulator config, so this cannot be recovered from the wire.
      Every metric output scales linearly with it.
- [ ] **Confirm `CAMERA_HFOV_DEG`** (currently an assumed 90 degrees).
      The FPV stream carries no intrinsics.
      Range and PnP scale inversely with the derived focal length.
- [ ] **Verify range monotonicity on a real frame sequence.**
      Only a single real frame has been available so far.
      Range ordering across the five gates in that frame is correct, and monotonic shrink on approach has been shown synthetically, but not yet on real consecutive frames.
- [ ] **Watch PnP yaw for noise.**
      See the jagged edges note below.
- [ ] Consider whether the controller wants a smoothed or filtered `vision_velocity`.
      It is currently a raw frame-to-frame difference.

### Tuning notes

These are the non-obvious findings, kept here because the code comments cannot carry the measurements.

**Saturation floor is 60, not the 120 that seemed reasonable.**
The gates' median saturation falls off hard with distance, because far gates wash out toward white:

| gate side | median S | at S>=120 | at S>=60 |
|---|---|---|---|
| 113px | 206 | clean | clean |
| 55px | 160 | 4 fragments | clean |
| 47px | 133 | 4 fragments | clean |
| 21px | 113 | no contour at all | clean |

A floor of 120 cut straight through the far gates.
Lowering it to 60 costs no purity, because saturation is not what rejects the distractors in the first place.
The city is grey (S is about 7), and the cyan line, blue panel and yellow HUD are all rejected on hue alone.
On the real frame, S>=60 leaves the background entirely black.
The only non-gate blob is 75px of a gate's own rail, far under the 200px area floor.

**No HUD masking is needed.**
`HUD_REGIONS_NORM` is empty.
Hue alone rejects every HUD element, measured at 0 mask pixels across the ACRO text, speed box, Hammertime bar, cyan tube and city regions.

**Hollowness must be measured on solid area, not raw contour area.**
`RETR_CCOMP`'s outer boundary of a hollow frame encloses the hole, so a raw `contourArea` over bounding box ratio is about 1.0 for a real gate.
Used directly it rejects every gate.
The child holes must be subtracted first.

**Hollowness is a preference, not a requirement.**
Past roughly 40px of gate side the mouth is closed up by blur and the morphological close.
Requiring a hole would blind the detector to exactly the gates it most needs to see coming.

**The open kernel is 3x3 while the close stays 5x5.**
A distant gate's rails are only a few pixels wide, and a 5x5 open erodes them away completely.
That loses the gate at about 40px of side, where a 3x3 open holds it to about 26px.
Since the S and V floors already produce a speckle-free mask, a large open costs real detection range and buys nothing.

**Jagged edges are mostly the gate's own texture, not threshold noise.**
The outer silhouette is clean, at 1 to 2px peak-to-peak.
The dots punched through the rails are the gate's white "AI-G7" lettering, which is low saturation and correctly falls below the floor.
The comb of teeth along the inner mouth edge is the gate's dotted trim, identifiable because the spacing is regular where noise would be irregular.
The inner edge is rougher than the outer because it borders the bright translucent blue panel, so blend pixels sweep through intermediate hues, whereas the outer edge is saturated red against near-black.
This does not affect the centroid, which averages over about 12000 pixels, or `range_m`, which comes from `minAreaRect`.
It can displace an `approxPolyDP` corner by a few pixels and add PnP yaw error.
If yaw proves noisy in flight, fit lines to the four rails for sub-pixel corners instead of trusting raw contour points.

### Design notes

**Publishing uses an atomic dict store, not a lock.**
`shared_data` is a plain dict with no existing locking convention, and the vision thread is its only writer.
A lock would only add contention against the 250Hz control loop.
Instead each snapshot is built off to the side and swapped in with a single dict store, which is atomic under the GIL.
Readers therefore see a whole frame's result, never a half-updated mix.

**`process_frame` takes the frame's `sim_time_ns`.**
The velocity estimate needs a timestamp, and wall-clock time would drift against sim time.

**Cost is about 1.5ms per frame**, roughly 4% of the 33ms budget at 30Hz, so the detector runs inline without stalling the receive loop.
Debug image dumping is the only slow step and is rate-limited.
Set `VISION_DEBUG_DIR` to dump source and mask pairs for offline inspection.

Command sequence that passed through the first gate: `uv run python .\scripts\target_tracker_probe.py --execute --hover-thrust 0.265 --climb-thrust-delta 0.015 --climb-duration 2.5 --forward-pitch-rate -0.05 --pitch-pulse-duration 0.5`
