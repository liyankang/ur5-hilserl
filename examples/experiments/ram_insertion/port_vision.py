"""Localize the USB socket with calibrated fixed and wrist camera views."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class PortVisionError(ValueError):
    """The calibration is invalid or the current frame has no reliable match."""


@dataclass(frozen=True)
class VisionEstimate:
    xy_m: np.ndarray
    pixel_xy: np.ndarray
    score: float
    second_score: Optional[float]


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise PortVisionError(f"{label} must be a finite number")
    return float(value)


def _numeric_array(value, shape, label):
    try:
        array = np.asarray(value)
    except ValueError as exc:
        raise PortVisionError(f"{label} must be a numeric array of shape {shape}") from exc
    if array.shape != shape or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise PortVisionError(f"{label} must be a finite numeric array of shape {shape}")
    return array.astype(np.float64)


def _integers(value, length, label):
    if not isinstance(value, list) or len(value) != length or any(type(item) is not int for item in value):
        raise PortVisionError(f"{label} must contain {length} integers")
    return tuple(value)


@dataclass(frozen=True)
class _TemplateMatcher:
    camera: str
    template_gray: np.ndarray
    frame_size_wh: tuple
    roi: tuple
    min_score: float
    min_peak_gap: float

    @classmethod
    def load(cls, config, directory, label):
        if not isinstance(config, dict):
            raise PortVisionError(f"{label} must be a JSON object")
        camera = config["camera"]
        if not isinstance(camera, str) or not camera.strip():
            raise PortVisionError(f"{label}.camera must name a configured camera")
        template_file = config["template"]
        if not isinstance(template_file, str) or not template_file.strip():
            raise PortVisionError(f"{label}.template must name an image file")
        path = directory / template_file
        try:
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        except (OSError, cv2.error) as exc:
            raise PortVisionError(f"Could not load {label} template {path}") from exc
        if image is None:
            raise PortVisionError(f"Could not decode {label} template {path}")
        template = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if min(template.shape) < 3 or np.std(template) < 1.0:
            raise PortVisionError(f"{label} template must contain a textured image patch")

        frame_size = _integers(config["frame_size_wh"], 2, f"{label}.frame_size_wh")
        roi = _integers(config["roi"], 4, f"{label}.roi")
        x, y, width, height = roi
        tpl_height, tpl_width = template.shape
        if x < 0 or y < 0 or width < tpl_width or height < tpl_height:
            raise PortVisionError(f"{label}.roi must fit the template and start inside the frame")
        if x + width > frame_size[0] or y + height > frame_size[1]:
            raise PortVisionError(f"{label}.roi is outside the calibrated frame")

        score = _number(config["min_score"], f"{label}.min_score")
        gap = _number(config["min_peak_gap"], f"{label}.min_peak_gap")
        if not 0.0 < score <= 1.0 or not 0.0 < gap <= 2.0:
            raise PortVisionError(f"{label} thresholds require 0 < min_score <= 1 and 0 < min_peak_gap <= 2")
        return cls(camera, template, frame_size, roi, score, gap)

    def match(self, frame_bgr):
        height, width = self.frame_size_wh[1], self.frame_size_wh[0]
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
                or frame_bgr.shape != (height, width, 3)):
            raise PortVisionError(f"{self.camera} requires a raw {width}x{height} BGR uint8 frame")

        x, y, width, height = self.roi
        search = cv2.cvtColor(frame_bgr[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
        scores = cv2.matchTemplate(search, self.template_gray, cv2.TM_CCOEFF_NORMED)
        if not np.isfinite(scores).all():
            raise PortVisionError(f"{self.camera} produced invalid template scores")
        _, score, _, (px, py) = cv2.minMaxLoc(scores)
        if score < self.min_score:
            raise PortVisionError(f"{self.camera} low-confidence match ({score:.3f} < {self.min_score:.3f})")

        tpl_height, tpl_width = self.template_gray.shape
        # Suppress nearby windows of the same feature before looking for a competing match.
        radius_x, radius_y = tpl_width // 2, tpl_height // 2
        scores[max(0, py - radius_y):py + radius_y + 1, max(0, px - radius_x):px + radius_x + 1] = -np.inf
        runner_up = float(scores.max())
        second_score = runner_up if math.isfinite(runner_up) else None
        if second_score is not None and score - second_score < self.min_peak_gap:
            raise PortVisionError(f"{self.camera} ambiguous match ({score:.3f} vs {second_score:.3f})")

        center_px = [x + px + (tpl_width - 1) / 2, y + py + (tpl_height - 1) / 2]
        return np.array(center_px), score, second_score


@dataclass(frozen=True)
class PortVision:
    """Fixed-camera global localization and wrist-camera local alignment."""

    approach_z_m: float
    _fixed: _TemplateMatcher
    _wrist: _TemplateMatcher
    _homography: np.ndarray
    _reference_pixel: np.ndarray
    _pixel_jacobian: np.ndarray

    @property
    def fixed_camera(self):
        return self._fixed.camera

    @property
    def wrist_camera(self):
        return self._wrist.camera

    @classmethod
    def load(cls, path):
        """Read the calibrated geometry and templates relative to its JSON file."""
        path = Path(path)
        try:
            config = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PortVisionError(f"Could not read vision calibration {path}: {exc}") from exc
        if not isinstance(config, dict):
            raise PortVisionError("Vision calibration must be a JSON object")

        try:
            approach_z_m = _number(config["approach_z_m"], "approach_z_m")
            fixed = _TemplateMatcher.load(config["fixed"], path.parent, "fixed")
            wrist = _TemplateMatcher.load(config["wrist"], path.parent, "wrist")
            homography = _numeric_array(config["fixed"]["pixel_to_base_homography"], (3, 3), "fixed homography")
            reference = _numeric_array(config["wrist"]["reference_pixel_xy"], (2,), "wrist reference pixel")
            jacobian = _numeric_array(config["wrist"]["pixel_per_meter_jacobian"], (2, 2), "wrist pixel Jacobian")
        except KeyError as exc:
            raise PortVisionError(f"Missing vision calibration field {exc}") from exc

        if fixed.camera == wrist.camera:
            raise PortVisionError("Fixed and wrist views must use separate cameras")
        if np.linalg.matrix_rank(homography) != 3 or np.linalg.matrix_rank(jacobian) != 2:
            raise PortVisionError("Fixed homography and wrist pixel Jacobian must be nonsingular")
        x, y, width, height = wrist.roi
        ref_x, ref_y = reference
        half_h, half_w = (np.asarray(wrist.template_gray.shape) - 1) / 2
        if not (x + half_w <= ref_x <= x + width - 1 - half_w
                and y + half_h <= ref_y <= y + height - 1 - half_h):
            raise PortVisionError("Wrist reference pixel must fit its whole template within the search ROI")
        return cls(approach_z_m, fixed, wrist, homography, reference, jacobian)

    def estimate_port_xy(self, frame_bgr):
        """Return absolute base XY for an inserted TCP aligned with the socket."""
        pixel, score, second = self._fixed.match(frame_bgr)
        homogeneous = self._homography @ np.r_[pixel, 1.0]
        denominator_scale = np.abs(self._homography[2]) @ np.r_[abs(pixel), 1.0]
        if abs(homogeneous[2]) <= 1e-12 * denominator_scale:
            raise PortVisionError("Fixed-camera homography is undefined at this pixel")
        xy = homogeneous[:2] / homogeneous[2]
        if not np.isfinite(xy).all():
            raise PortVisionError("Fixed-camera homography returned invalid base XY")
        return VisionEstimate(xy, pixel, score, second)

    def wrist_correction(self, frame_bgr):
        """Return the required TCP translation at the calibrated height and orientation."""
        pixel, score, second = self._wrist.match(frame_bgr)
        # Calibration columns measure feature pixel motion per positive TCP-axis motion.
        try:
            xy = np.linalg.solve(self._pixel_jacobian, self._reference_pixel - pixel)
        except np.linalg.LinAlgError as exc:
            raise PortVisionError("Could not solve for a wrist correction") from exc
        if not np.isfinite(xy).all():
            raise PortVisionError("Wrist-camera calibration returned an invalid correction")
        return VisionEstimate(xy, pixel, score, second)
