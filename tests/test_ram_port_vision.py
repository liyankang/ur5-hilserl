import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from examples.calibrate_ram_port_vision import fit
from examples.experiments.ram_insertion.port_vision import PortVision, PortVisionError


class RamPortVisionTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.patch = np.random.default_rng(21).integers(0, 256, (9, 11, 3), dtype=np.uint8)

    def frame(self, *centers):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        for x, y in centers:
            frame[y - 4:y + 5, x - 5:x + 6] = self.patch
        return frame

    def make_vision(self):
        fixed_origin = (30, 30)
        wrist_origin = (80, 60)
        self.assertTrue(cv2.imwrite(str(self.path / "fixed.png"), self.frame(fixed_origin)))
        self.assertTrue(cv2.imwrite(str(self.path / "wrist.png"), self.frame(wrist_origin)))
        roi = [0, 0, 160, 120]
        fixed = {
            "camera": "global_2",
            "image": "fixed.png",
            "template_roi": [25, 26, 11, 9],
            "roi": roi,
            "samples": [
                {"pixel_xy": [x, y], "base_xy_m": [0.06 + x * 0.0004, -0.68 + y * 0.0004]}
                for x, y in (fixed_origin, (120, 30), (30, 90), (120, 90), (60, 60))
            ],
        }
        wrist = {
            "camera": "wrist",
            "image": "wrist.png",
            "template_roi": [75, 56, 11, 9],
            "roi": roi,
            "reference_tcp_xy_m": [0.1, -0.65],
            "samples": [
                {"pixel_xy": [80 + dx, 60 + dy], "tcp_xy_m": [0.1 + x, -0.65 + y]}
                for x, y, dx, dy in (
                    (0.005, 0.0, -10, 3),
                    (-0.005, 0.0, 10, -3),
                    (0.0, 0.005, 2, 9),
                    (0.0, -0.005, -2, -9),
                )
            ],
        }
        measurements = self.path / "measurements.json"
        measurements.write_text(json.dumps({"approach_z_m": -0.10, "fixed": fixed, "wrist": wrist}))
        calibration = self.path / "calibration.json"
        fit(measurements, calibration)
        return PortVision.load(calibration)

    def test_fit_and_detect_a_previously_unseen_fixed_position_and_wrist_offset(self):
        vision = self.make_vision()

        location = vision.estimate_port_xy(self.frame((70, 40)))
        correction = vision.wrist_correction(self.frame((68, 54)))

        np.testing.assert_allclose(location.xy_m, [0.088, -0.664], atol=1e-7)
        np.testing.assert_allclose(location.pixel_xy, [70, 40])
        np.testing.assert_allclose(correction.xy_m, [-0.005, 0.005], atol=1e-7)

    def test_rejects_ambiguous_matches_and_uncalibrated_frame_size(self):
        vision = self.make_vision()

        with self.assertRaisesRegex(PortVisionError, "ambiguous"):
            vision.estimate_port_xy(self.frame((40, 50), (110, 65)))
        with self.assertRaisesRegex(PortVisionError, "raw"):
            vision.wrist_correction(self.frame((80, 60))[:60])
