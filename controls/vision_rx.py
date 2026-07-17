import os
import socket
import struct
import threading
import time

import cv2
import numpy as np

# Modify these properties if you want to run the server remotely for example
SIM_SERVER_UDP_IP = "0.0.0.0"
SIM_SERVER_UDP_PORT = 5600

# --------------------------------------------------------------------------------------
# GATE APPEARANCE
# --------------------------------------------------------------------------------------
# The gate is the only strongly saturated red-orange object in the scene (~RGB 240,60,40,
# which is H~3-5, S~235, V~240 in OpenCV's 0-179 hue convention). Red wraps around the
# hue circle, so it needs two bands OR'd together.
#
# The background is a near-black, near-desaturated wireframe city, so the S/V floors can
# be aggressive. Aggressive floors are what buy a clean mask: the cyan racing line, the
# translucent blue panel in the gate mouth and the yellow/white HUD text are all far from
# red in hue, and everything else in frame is too dark or too grey to clear V_MIN/S_MIN.
HUE_LO_RANGE = (0, 10)      # low side of the red wrap
HUE_HI_RANGE = (170, 179)   # high side of the red wrap (OpenCV hue maxes at 179)

# Tuned against a real 1918x1078 frame with five gates at different ranges, not guessed.
# Measured median saturation of the gates' red pixels falls off hard with distance:
#   near (113px side) S=206 | 55px S=160 | 47px S=133 | 21px S=113
# so an S floor of 120 cuts through the middle of the far gates: it kept only 40% of the
# farthest gate's red pixels and left it with no substantial contour at all, while
# shattering the mid gates into 4-5 fragments. Dropping the floor to 60 resolves every
# gate into a single clean blob and still admits no real background speckle - the city is
# grey (S~7) and the cyan line / blue panel / yellow HUD are all rejected on hue alone, so
# S is not what protects us from them. On this frame S>=60 leaves the background totally
# black; the only non-gate blob is 75px of a gate's own rail. Value is not the limiter
# (the gates' V stays >=141 at the 10th percentile), so its floor stays at 100.
SAT_MIN = 60
VAL_MIN = 100

# Close knits the hollow frame's rails and the JPEG-subsampled red edges into one
# component, so it wants the bigger kernel. The open only has to kill speckle, and it is
# deliberately smaller: a distant gate's rails are only a few pixels wide, and a 5x5 open
# erodes them away completely, losing the gate at ~40px of side where a 3x3 open holds it
# to ~26px. Since the S/V floors already produce a speckle-free mask, a big open costs
# real detection range and buys nothing. Revisit if real frames show speckle.
CLOSE_KERNEL_SIZE = (5, 5)
OPEN_KERNEL_SIZE = (3, 3)

# --------------------------------------------------------------------------------------
# CANDIDATE FILTERING
# --------------------------------------------------------------------------------------
MIN_CONTOUR_AREA_PX = 200.0    # below this a "gate" is too far away to steer on reliably
ASPECT_RATIO_TOLERANCE = 0.45  # the gate is square, so minAreaRect aspect should be ~1.0

# Hollow-ness. Note this is measured on the SOLID area (outer contour area minus its
# holes), not on cv2.contourArea of the outer contour: RETR_CCOMP's outer boundary of a
# hollow frame encloses the hole, so its raw contourArea/rect ratio is ~1.0 for a gate and
# rejects every real gate. Rails only, so a true gate fills well under half its box.
MAX_FILL_RATIO = 0.80

# A gate far enough away has its mouth closed up by JPEG blur + the morphological close,
# so it arrives as a solid blob with no hole and fill ~1.0. Requiring hollow-ness outright
# would blind us to exactly the gates we most need to see coming. So the fill test only
# applies once the gate is big enough for its mouth to actually survive as a hole.
# Measured on synthetic frames: the mouth is gone by ~40px of gate side and solid by ~90px,
# hence 60 with margin. Worth re-checking against real frames, since the exact crossover
# depends on the sim's rail thickness and JPEG quality.
HOLE_RESOLVABLE_SIDE_PX = 60.0

# --------------------------------------------------------------------------------------
# CAMERA / GATE GEOMETRY
#
# !!! UNCALIBRATED ASSUMPTIONS - see notes below !!!
#
# Gate dimensions are no longer published in telemetry (they are nulled by the current
# simulator config, see mavlink_rx.on_track_data), and the FPV stream carries no
# intrinsics, so range/PnP depend on these two numbers being right. They are estimates,
# not measurements. Every metric output (range_m, gate_body_pos, pnp_*, vision_velocity)
# scales linearly with GATE_OUTER_WIDTH_M and inversely with the focal length, so both
# must be checked against ground truth before the controller trusts absolute distances.
# Bearing (the centroid offset) is unaffected by GATE_OUTER_WIDTH_M and is the trustworthy
# part of this estimate until calibration happens.
# --------------------------------------------------------------------------------------
GATE_OUTER_WIDTH_M = 1.5      # ASSUMPTION: outer edge-to-edge width of the square gate
CAMERA_HFOV_DEG = 90.0        # ASSUMPTION: horizontal field of view of the FPV camera

# --------------------------------------------------------------------------------------
# HUD MASKING
#
# If HUD elements are burned into the camera stream, list their screen regions here as
# normalized (x0, y0, x1, y1) in [0,1] and they are zeroed out of the mask before contour
# finding. Hue alone should reject cyan/yellow/white HUD text, so this stays empty until
# a real frame shows red-ish HUD pixels surviving the threshold.
# --------------------------------------------------------------------------------------
HUD_REGIONS_NORM = []

# Set VISION_DEBUG_DIR to dump {frame}_src.png / {frame}_mask.png for offline inspection.
DEBUG_DUMP_DIR = os.environ.get("VISION_DEBUG_DIR")
DEBUG_DUMP_EVERY_N = int(os.environ.get("VISION_DEBUG_EVERY_N", "15"))

# --------------------------------------------------------------------------------------
# TARGET TRACKING
# --------------------------------------------------------------------------------------
TRACK_MAX_MISSED_FRAMES = 3
TRACK_MAX_SIZE_RATIO = 2.5
TRACK_MAX_CENTER_DISTANCE_FRAC = 0.12
TRACK_EDGE_MARGIN_FRAC = 0.02


def _order_corners(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left.

    Consistent ordering is what makes the PnP rvec mean the same thing frame to frame.
    """
    pts = pts.astype(np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(4)
    return np.array([
        pts[np.argmin(s)],  # top-left     has the smallest x+y
        pts[np.argmin(d)],  # top-right    has the smallest y-x
        pts[np.argmax(s)],  # bottom-right has the largest  x+y
        pts[np.argmax(d)],  # bottom-left  has the largest  y-x
    ], dtype=np.float32)


def _bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


class GateTracker:
    """Keep one gate target stable across competing contours and brief dropouts."""

    def __init__(self, max_missed_frames=TRACK_MAX_MISSED_FRAMES):
        self.max_missed_frames = max_missed_frames
        self.track_id = 0
        self.current = None
        self.missed_frames = 0
        self.gate_index = None

    def reset(self):
        self.current = None
        self.missed_frames = 0

    def update(self, candidates, frame_shape, gate_index=None):
        candidates = sorted(candidates, key=lambda item: item["area_px"], reverse=True)
        gate_changed = (
            gate_index is not None
            and self.gate_index is not None
            and gate_index != self.gate_index
        )
        if gate_index is not None:
            self.gate_index = gate_index
        if gate_changed:
            self.reset()

        usable = [
            candidate for candidate in candidates
            if not self._is_exit_fragment(candidate, frame_shape)
        ]

        switched = False
        switch_reason = None
        association_score = None

        if self.current is None:
            selected = usable[0] if usable else None
            if selected is not None:
                self.track_id += 1
                self.current = selected
                self.missed_frames = 0
                switched = True
                switch_reason = "gate_index_changed" if gate_changed else "acquired"
        else:
            selected, association_score = self._associate(usable, frame_shape)
            if selected is not None:
                self.current = selected
                self.missed_frames = 0
            else:
                self.missed_frames += 1
                # Keep the active-gate lock until telemetry advances the gate index.
                # Selecting the largest remaining contour here can redirect control to
                # a later gate during the short delay between crossing and telemetry.

        confidence = 0.0
        if self.current is not None:
            confidence = max(
                0.0,
                1.0 - self.missed_frames / (self.max_missed_frames + 1),
            )
            if association_score is not None:
                confidence *= max(0.0, 1.0 - association_score)

        detection = self.current if self.current is not None and self.missed_frames == 0 else None
        return detection, {
            "track_id": self.track_id if self.current is not None else None,
            "candidate_count": len(candidates),
            "tracking_confidence": confidence,
            "tracking_missed_frames": self.missed_frames,
            "track_switched": switched,
            "track_switch_reason": switch_reason,
            "association_score": association_score,
            "rejected_edge_fragments": len(candidates) - len(usable),
            "candidates": tuple(self._candidate_summary(item) for item in candidates),
        }

    def _associate(self, candidates, frame_shape):
        h, w = frame_shape[:2]
        diagonal = max((w * w + h * h) ** 0.5, 1.0)
        previous_area = max(self.current["area_px"], 1.0)
        best = None
        best_score = None

        for candidate in candidates:
            area = max(candidate["area_px"], 1.0)
            size_ratio = max(area / previous_area, previous_area / area)
            if size_ratio > TRACK_MAX_SIZE_RATIO:
                continue

            px, py = self.current["centroid"]
            cx, cy = candidate["centroid"]
            center_distance = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 / diagonal
            overlap = _bbox_iou(self.current["bbox"], candidate["bbox"])
            if center_distance > TRACK_MAX_CENTER_DISTANCE_FRAC and overlap == 0.0:
                continue

            size_penalty = abs(np.log(area / previous_area)) / np.log(TRACK_MAX_SIZE_RATIO)
            score = 0.55 * (center_distance / TRACK_MAX_CENTER_DISTANCE_FRAC)
            score += 0.25 * size_penalty
            score += 0.20 * (1.0 - overlap)
            if best_score is None or score < best_score:
                best, best_score = candidate, float(score)

        return best, best_score

    @staticmethod
    def _is_exit_fragment(candidate, frame_shape):
        h, w = frame_shape[:2]
        x, y, bw, bh = candidate["bbox"]
        margin_x = w * TRACK_EDGE_MARGIN_FRAC
        margin_y = h * TRACK_EDGE_MARGIN_FRAC
        touches_edge = (
            x <= margin_x
            or y <= margin_y
            or x + bw >= w - margin_x
            or y + bh >= h - margin_y
        )
        cx, cy = candidate["centroid"]
        off_center = abs(cx - w / 2.0) > 0.20 * w or abs(cy - h / 2.0) > 0.20 * h
        return touches_edge and off_center

    @staticmethod
    def _candidate_summary(candidate):
        return {
            "bbox": candidate["bbox"],
            "centroid": candidate["centroid"],
            "area_px": candidate["area_px"],
            "has_hole": candidate["has_hole"],
        }


class FrameSequenceGuard:
    """Ignore repeated/late frames and identify real simulator reset epochs."""

    def __init__(self):
        self.last_frame_id = None
        self.reset_epoch = None

    def accept(self, frame_id, reset_epoch=None):
        epoch_changed = reset_epoch is not None and reset_epoch != self.reset_epoch
        if epoch_changed:
            self.reset_epoch = reset_epoch
            self.last_frame_id = None

        if self.last_frame_id is not None and frame_id <= self.last_frame_id:
            return False, epoch_changed

        self.last_frame_id = frame_id
        return True, epoch_changed


class VisionRX:

    def __init__(self, data, bind_ip=SIM_SERVER_UDP_IP, bind_port=SIM_SERVER_UDP_PORT):
        self.data = data
        self.bind_ip = bind_ip
        self.bind_port = bind_port
        self.thread = threading.Thread(
            target=self._vision_loop,
            daemon=False
        )

        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSE_KERNEL_SIZE)
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPEN_KERNEL_SIZE)
        self._intrinsics = None          # (K, dist), built lazily once frame size is known
        self._object_points = None       # gate corners in gate-local metres
        self._prev = None                # (t_s, body_pos) of the last successful detection
        self._frame_count = 0
        self._frame_sequence = FrameSequenceGuard()
        self._tracker = GateTracker()

        self.is_running = True
        self.thread.start()

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def _vision_loop(self):
        header_format = "<IHHIIQ"
        header_sz = struct.calcsize(header_format)
        frames = {}  # frame_id -> received associated frame data

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.bind_ip, self.bind_port))
        sock.settimeout(0.2)
        print("Listening for camera frames...")

        while self.is_running:
            try:
                packet, addr = sock.recvfrom(65536)  # max UDP size
            except socket.timeout:
                continue

            header = packet[:header_sz]
            payload = packet[header_sz:]

            # frame_id - identifier for this vision frame
            # chunk_id - identifier for this chunk packet of data of this frame
            # total_chunks - total number of chunk packets that make up this frame
            # jpeg_size - full size of jpeg data
            # payload_size - size of this packet
            # sim_time_ns - frame's epoch timestamp in ns on the server
            frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = struct.unpack(header_format, header)

            if frame_id not in frames:
                frames[frame_id] = {
                    "chunks": {},
                    "total": total_chunks,
                    "size": jpeg_size,
                    "time": sim_time_ns
                }

            frames[frame_id]["chunks"][chunk_id] = payload

            # Check if frame is complete
            if len(frames[frame_id]["chunks"]) == total_chunks:
                jpeg_bytes = bytearray()

                frame_complete = True
                for i in range(total_chunks):
                    if i not in frames[frame_id]["chunks"]:
                        print('Missing packet %s in frame %s' % (i, frame_id,))
                        frame_complete = False
                        continue
                    jpeg_bytes.extend(frames[frame_id]["chunks"][i])

                if not frame_complete:
                    del frames[frame_id]
                    continue

                img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if image is not None:
                    self.process_frame(frame_id, image, frames[frame_id]["time"])
                else:
                    print(f"Failed to decode frame: {frame_id}")

                del frames[frame_id]

    # ----------------------------------------------------------------------------------
    # Detection
    # ----------------------------------------------------------------------------------
    def process_frame(self, frame_id, img, sim_time_ns=None):
        """Detect the nearest red gate and publish the result to shared data.

        Runs inline on the receive loop. The whole pipeline is a few milliseconds on a
        frame this size, which is well inside the ~33ms frame budget at 30Hz, so it does
        not stall packet reassembly. Anything genuinely slow (debug image writes) is
        rate-limited and must stay that way.
        """
        # `is not None`, not truthiness: sim_time_ns == 0 is a real timestamp at sim boot,
        # and falling back to wall-clock for it would corrupt the first velocity delta.
        t_s = (sim_time_ns / 1e9) if sim_time_ns is not None else time.time()
        self._frame_count += 1

        race_status = self.data.get("race_status") or {}
        accepted, reset_epoch_changed = self._frame_sequence.accept(
            frame_id,
            race_status.get("race_start_boot_time_ms"),
        )
        if reset_epoch_changed:
            self._tracker.reset()
            self._prev = None
        if not accepted:
            return

        mask = self._build_mask(img)
        candidates = self._find_gate_candidates(mask, img.shape, t_s)
        detection, tracking = self._tracker.update(
            candidates,
            img.shape,
            race_status.get("active_gate_index"),
        )
        if tracking["track_switched"]:
            self._prev = None
        if detection is not None:
            detection = dict(detection)
            detection["vision_velocity"] = self._estimate_velocity(
                detection["gate_body_pos"], t_s
            )

        self._publish(frame_id, t_s, detection, tracking, img.shape)
        self._maybe_dump_debug(frame_id, img, mask)

    def _build_mask(self, img):
        """Threshold the one narrow red hue band. Target: gate solid white, all else black."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lo = cv2.inRange(hsv, (HUE_LO_RANGE[0], SAT_MIN, VAL_MIN), (HUE_LO_RANGE[1], 255, 255))
        hi = cv2.inRange(hsv, (HUE_HI_RANGE[0], SAT_MIN, VAL_MIN), (HUE_HI_RANGE[1], 255, 255))
        mask = cv2.bitwise_or(lo, hi)

        h, w = mask.shape[:2]
        for (x0, y0, x1, y1) in HUD_REGIONS_NORM:
            mask[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 0

        # Close first so the hollow frame's rails and the JPEG-subsampled red edges knit
        # into one connected component, then open to kill what speckle survives.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)
        return mask

    def _find_gate_candidates(self, mask, shape, t_s):
        """Return every valid gate candidate, ordered from largest to smallest."""
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return []
        hierarchy = hierarchy[0]

        candidates = []
        for i, contour in enumerate(contours):
            # RETR_CCOMP puts outer boundaries at the top level; holes are their children.
            if hierarchy[i][3] != -1:
                continue

            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA_PX:
                continue

            (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
            if rw <= 0 or rh <= 0:
                continue
            # Always >=1, so the test does not depend on which side minAreaRect happened to
            # call "width" - that assignment flips with the gate's rotation angle.
            aspect = max(rw, rh) / min(rw, rh)
            if aspect - 1.0 > ASPECT_RATIO_TOLERANCE:
                continue

            # Subtract the holes to get the area actually painted red: a hollow frame is
            # rails only, a red wall or billboard is filled.
            hole_area = 0.0
            child = hierarchy[i][2]
            while child != -1:
                hole_area += cv2.contourArea(contours[child])
                child = hierarchy[child][0]
            has_hole = hierarchy[i][2] != -1
            fill_ratio = (area - hole_area) / (rw * rh)

            # Only demand hollow-ness once the mouth is big enough to survive as a hole.
            if min(rw, rh) >= HOLE_RESOLVABLE_SIDE_PX and fill_ratio > MAX_FILL_RATIO:
                continue

            measurement = self._measure(contour, has_hole, shape, t_s)
            if measurement is not None:
                candidates.append(measurement)

        return sorted(candidates, key=lambda item: item["area_px"], reverse=True)

    def _measure(self, contour, has_hole, shape, t_s):
        h, w = shape[:2]
        K, dist = self._get_intrinsics(w, h)
        fx, fy = K[0, 0], K[1, 1]
        ppx, ppy = K[0, 2], K[1, 2]

        x, y, bw, bh = cv2.boundingRect(contour)

        m = cv2.moments(contour)
        if m["m00"] <= 0:
            return None
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]

        # Range from the pinhole model on the gate's known outer width. minAreaRect is
        # rotation-invariant, so it beats the axis-aligned box when the gate is banked.
        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        width_px = 0.5 * (rw + rh)
        if width_px <= 1.0:
            return None
        range_m = (fx * GATE_OUTER_WIDTH_M) / width_px

        # Body frame is FRD: x forward, y right, z down. The camera looks down body +x,
        # so camera x (right) maps to body y and camera y (down) maps to body z.
        body_pos = (
            float(range_m),
            float((cx - ppx) * range_m / fx),
            float((cy - ppy) * range_m / fy),
        )

        pnp_ok, pnp_rvec = self._solve_pnp(contour, K, dist)
        return {
            "bbox": (int(x), int(y), int(bw), int(bh)),
            "centroid": (float(cx), float(cy)),
            "area_px": float(cv2.contourArea(contour)),
            "has_hole": bool(has_hole),
            "range_m": float(range_m),
            "gate_body_pos": body_pos,
            "pnp_ok": pnp_ok,
            "pnp_rvec": pnp_rvec,
            "vision_velocity": None,
        }

    def _solve_pnp(self, contour, K, dist):
        """Gate-normal pose from the four outer corners, for yaw alignment."""
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            return False, None

        image_points = _order_corners(approx)
        ok, rvec, _ = cv2.solvePnP(
            self._object_points, image_points, K, dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return False, None
        return True, tuple(float(v) for v in rvec.reshape(3))

    def _estimate_velocity(self, body_pos, t_s):
        """Body-frame velocity from frame-to-frame motion of the gate.

        The gate is static, so the vehicle's velocity is the negation of the gate's
        apparent motion in the body frame. Returns None on the first detection and
        whenever the gate was lost in between, since differencing across a gap would
        smear two different gates into a bogus velocity.
        """
        prev = self._prev
        self._prev = (t_s, body_pos)
        if prev is None:
            return None

        dt = t_s - prev[0]
        if dt <= 0 or dt > 0.25:  # >0.25s means we lost the gate; don't difference across it
            return None
        return tuple(-(body_pos[i] - prev[1][i]) / dt for i in range(3))

    def _get_intrinsics(self, w, h):
        if self._intrinsics is None:
            fx = (w / 2.0) / np.tan(np.deg2rad(CAMERA_HFOV_DEG) / 2.0)
            K = np.array([
                [fx, 0.0, w / 2.0],
                [0.0, fx, h / 2.0],  # square pixels: fy == fx
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            self._intrinsics = (K, np.zeros((4, 1), dtype=np.float64))

            half = GATE_OUTER_WIDTH_M / 2.0
            # Matches _order_corners: TL, TR, BR, BL in the gate's own plane.
            self._object_points = np.array([
                [-half, -half, 0.0],
                [half, -half, 0.0],
                [half, half, 0.0],
                [-half, half, 0.0],
            ], dtype=np.float32)
        return self._intrinsics

    # ----------------------------------------------------------------------------------
    # Publishing
    # ----------------------------------------------------------------------------------
    def _publish(self, frame_id, t_s, detection, tracking, frame_shape):
        """Publish an immutable snapshot under a single atomic dict assignment.

        shared_data is a plain dict with no established locking convention and this is its
        only writer, so a lock would only add contention against the 250Hz control loop.
        Instead the snapshot is built off to the side and swapped in with one dict
        store, which is atomic under the GIL. A reader therefore sees either the whole
        previous frame's result or the whole new one, never a half-updated mix.
        """
        snapshot = {
            "frame_id": frame_id,
            "timestamp_s": t_s,
            "detected": detection is not None,
            "frame_size": (int(frame_shape[1]), int(frame_shape[0])),
            **tracking,
        }
        if detection is not None:
            snapshot.update(detection)
        else:
            # Explicit no-detection state, so the controller can tell a lost gate from a
            # stale one instead of steering at a gate that left the frame seconds ago.
            snapshot.update({
                "bbox": None,
                "centroid": None,
                "range_m": None,
                "gate_body_pos": None,
                "pnp_ok": False,
                "pnp_rvec": None,
                "vision_velocity": None,
            })
            self._prev = None

        self.data["gate"] = snapshot

    def _maybe_dump_debug(self, frame_id, img, mask):
        if not DEBUG_DUMP_DIR or (self._frame_count % DEBUG_DUMP_EVERY_N):
            return
        os.makedirs(DEBUG_DUMP_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(DEBUG_DUMP_DIR, f"{frame_id:06d}_src.png"), img)
        cv2.imwrite(os.path.join(DEBUG_DUMP_DIR, f"{frame_id:06d}_mask.png"), mask)
