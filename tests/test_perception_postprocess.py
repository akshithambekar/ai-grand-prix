from __future__ import annotations

import unittest

import numpy as np

from perception.postprocess import extract_detection_from_mask


class PostprocessTests(unittest.TestCase):
    def test_extract_detection_from_mask_returns_bbox_and_corners(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[4:15, 6:14] = 1

        detection = extract_detection_from_mask(mask, confidence=0.9, roi_offset_xy=(10, 20))

        self.assertIsNotNone(detection)
        assert detection is not None
        self.assertEqual(detection.roi_bbox_xyxy, (6, 4, 14, 15))
        self.assertEqual(detection.full_frame_bbox_xyxy, (16, 24, 24, 35))
        self.assertEqual(len(detection.ordered_corners), 4)
        self.assertGreater(detection.mask_summary["mask_area_px"], 0.0)


if __name__ == "__main__":
    unittest.main()
