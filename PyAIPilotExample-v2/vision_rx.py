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


class VisionRX:

    def __init__(self, data):
        self.data = data
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
        sock.bind((SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT))
        print("Listening for camera frames...")

        while self.is_running:
            packet, addr = sock.recvfrom(65536)  # max UDP size

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

        mask = self._build_mask(img)
        detection = self._find_gate(mask, img.shape, t_s)

        self._publish(frame_id, t_s, detection)
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

    def _find_gate(self, mask, shape, t_s):
        """Pick the nearest gate-shaped contour, or return None."""
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return None
        hierarchy = hierarchy[0]

        best = None
        best_area = 0.0
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

            # Largest survivor is the nearest gate, and the one to fly.
            if area > best_area:
                best_area = area
                best = (contour, has_hole)

        if best is None:
            return None
        return self._measure(best[0], best[1], shape, t_s)

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
        velocity = self._estimate_velocity(body_pos, t_s)

        return {
            "bbox": (int(x), int(y), int(bw), int(bh)),
            "centroid": (float(cx), float(cy)),
            "area_px": float(cv2.contourArea(contour)),
            "has_hole": bool(has_hole),
            "range_m": float(range_m),
            "gate_body_pos": body_pos,
            "pnp_ok": pnp_ok,
            "pnp_rvec": pnp_rvec,
            "vision_velocity": velocity,
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
    def _publish(self, frame_id, t_s, detection):
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
