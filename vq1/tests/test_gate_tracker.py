import unittest

from controls.vision_rx import FrameSequenceGuard, GATE_OUTER_WIDTH_M, GateTracker, VisionRX


FRAME_SHAPE = (1080, 1920, 3)


def gate(cx, cy, side, area=None, has_hole=True):
    return {
        "bbox": (cx - side // 2, cy - side // 2, side, side),
        "centroid": (float(cx), float(cy)),
        "area_px": float(area if area is not None else side * side),
        "has_hole": has_hole,
        "range_m": 1.0,
        "gate_body_pos": (1.0, 0.0, 0.0),
        "pnp_ok": False,
        "pnp_rvec": None,
        "vision_velocity": None,
    }


class GateTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = GateTracker(max_missed_frames=3)

    def test_stable_approach_keeps_track_id(self):
        selected, state = self.tracker.update([gate(960, 540, 80)], FRAME_SHAPE, 0)
        track_id = state["track_id"]

        for side in (88, 96, 108):
            selected, state = self.tracker.update(
                [gate(962, 542, side)], FRAME_SHAPE, 0
            )
            self.assertIsNotNone(selected)
            self.assertEqual(state["track_id"], track_id)
            self.assertFalse(state["track_switched"])

    def test_larger_distant_candidate_does_not_steal_track(self):
        _, state = self.tracker.update([gate(700, 500, 80)], FRAME_SHAPE, 0)
        track_id = state["track_id"]
        selected, state = self.tracker.update(
            [gate(1300, 500, 120), gate(704, 502, 86)], FRAME_SHAPE, 0
        )

        self.assertEqual(selected["centroid"], (704.0, 502.0))
        self.assertEqual(state["track_id"], track_id)

    def test_brief_dropout_reacquires_same_track(self):
        _, state = self.tracker.update([gate(960, 540, 80)], FRAME_SHAPE, 0)
        track_id = state["track_id"]

        for missed in range(1, 4):
            selected, state = self.tracker.update([], FRAME_SHAPE, 0)
            self.assertIsNone(selected)
            self.assertEqual(state["track_id"], track_id)
            self.assertEqual(state["tracking_missed_frames"], missed)

        selected, state = self.tracker.update([gate(965, 543, 95)], FRAME_SHAPE, 0)
        self.assertIsNotNone(selected)
        self.assertEqual(state["track_id"], track_id)

    def test_unrelated_candidate_cannot_replace_lost_active_gate(self):
        _, state = self.tracker.update([gate(960, 540, 300)], FRAME_SHAPE, 0)
        track_id = state["track_id"]

        for _ in range(self.tracker.max_missed_frames + 2):
            selected, state = self.tracker.update(
                [gate(1100, 700, 40)], FRAME_SHAPE, 0
            )

        self.assertIsNone(selected)
        self.assertEqual(state["track_id"], track_id)
        self.assertFalse(state["track_switched"])

    def test_lost_active_gate_can_reassociate_after_extended_dropout(self):
        _, state = self.tracker.update([gate(960, 540, 100)], FRAME_SHAPE, 0)
        track_id = state["track_id"]

        for _ in range(self.tracker.max_missed_frames + 2):
            selected, _ = self.tracker.update([], FRAME_SHAPE, 0)
            self.assertIsNone(selected)

        selected, state = self.tracker.update([gate(965, 543, 110)], FRAME_SHAPE, 0)

        self.assertIsNotNone(selected)
        self.assertEqual(state["track_id"], track_id)
        self.assertFalse(state["track_switched"])

    def test_off_center_edge_fragment_is_rejected(self):
        self.tracker.update([gate(960, 540, 100)], FRAME_SHAPE, 0)
        fragment = gate(30, 500, 100, area=20_000)
        selected, state = self.tracker.update(
            [fragment, gate(965, 542, 110)], FRAME_SHAPE, 0
        )

        self.assertEqual(selected["centroid"], (965.0, 542.0))
        self.assertEqual(state["rejected_edge_fragments"], 1)

    def test_gate_index_advance_creates_new_track(self):
        _, state = self.tracker.update([gate(960, 540, 100)], FRAME_SHAPE, 0)
        old_track = state["track_id"]
        selected, state = self.tracker.update([gate(1000, 600, 50)], FRAME_SHAPE, 1)

        self.assertIsNotNone(selected)
        self.assertNotEqual(state["track_id"], old_track)
        self.assertTrue(state["track_switched"])
        self.assertEqual(state["track_switch_reason"], "gate_index_changed")


class FrameSequenceGuardTests(unittest.TestCase):
    def test_duplicate_and_late_frames_are_ignored(self):
        guard = FrameSequenceGuard()

        self.assertEqual(guard.accept(100, 4000), (True, True))
        self.assertEqual(guard.accept(100, 4000), (False, False))
        self.assertEqual(guard.accept(99, 4000), (False, False))
        self.assertEqual(guard.accept(101, 4000), (True, False))

    def test_new_race_epoch_accepts_restarted_frame_ids(self):
        guard = FrameSequenceGuard()
        guard.accept(500, 4000)

        self.assertEqual(guard.accept(1, 9000), (True, True))
        self.assertEqual(guard.accept(2, 9000), (True, False))


class VisionGeometryTests(unittest.TestCase):
    def test_restored_gate_dimensions_override_assumption(self):
        receiver = VisionRX.__new__(VisionRX)
        receiver.data = {
            "active_gate_state": {"valid": True, "width_m": 2.5, "height_m": 3.0}
        }
        self.assertEqual(receiver._active_gate_dimensions(), (2.5, 3.0, "track"))

    def test_invalid_track_geometry_uses_documented_fallback(self):
        receiver = VisionRX.__new__(VisionRX)
        receiver.data = {"active_gate_state": {"valid": False}}
        self.assertEqual(
            receiver._active_gate_dimensions(),
            (GATE_OUTER_WIDTH_M, GATE_OUTER_WIDTH_M, "assumed"),
        )


if __name__ == "__main__":
    unittest.main()
