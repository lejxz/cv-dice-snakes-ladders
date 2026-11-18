"""Dual-resolution preprocessing for fast die localization and crisp pip recognition."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import PreprocessingConfig, ROIConfig


@dataclass
class PreprocessedFrame:
    """Synchronized full-resolution and downscaled grayscale views of a single frame."""

    full_frame: np.ndarray
    roi: ROIConfig
    roi_bgr: np.ndarray
    roi_gray_raw: np.ndarray
    recognition_gray: np.ndarray
    detection_gray: np.ndarray
    scale_x: float
    scale_y: float
    inv_scale_x: float
    inv_scale_y: float
    processing_time_ms: float

    def detection_to_roi(self, x_det: float, y_det: float) -> Tuple[float, float]:
        """Map (x, y) from downscaled detection_gray coordinates to full-res ROI coordinates."""
        return (x_det * self.inv_scale_x, y_det * self.inv_scale_y)

    def detection_to_full(self, x_det: float, y_det: float) -> Tuple[float, float]:
        """Map (x, y) from downscaled detection_gray coordinates to full-frame coordinates."""
        rx, ry = self.detection_to_roi(x_det, y_det)
        return (rx + self.roi.x, ry + self.roi.y)

    def scale_rect_to_roi(
        self, rect: Tuple[Tuple[float, float], Tuple[float, float], float]
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """Scale a cv2.minAreaRect ((cx, cy), (w, h), angle) from detection space to ROI space."""
        (cx, cy), (w, h), angle = rect
        return (
            (cx * self.inv_scale_x, cy * self.inv_scale_y),
            (w * self.inv_scale_x, h * self.inv_scale_y),
            angle,
        )

    def scale_rect_to_full(
        self, rect: Tuple[Tuple[float, float], Tuple[float, float], float]
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """Scale a cv2.minAreaRect from detection space to full-frame coordinates."""
        (rx, ry), (rw, rh), angle = self.scale_rect_to_roi(rect)
        return ((rx + self.roi.x, ry + self.roi.y), (rw, rh), angle)


def _normalize_odd_ksize(ksize: Tuple[int, int]) -> Tuple[int, int]:
    kx = max(1, int(ksize[0]) | 1)
    ky = max(1, int(ksize[1]) | 1)
    return (kx, ky)


class FramePreprocessor:
    """Produces downscaled detection and full-resolution recognition grayscale streams."""

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()
        if not (0.1 <= self.config.detection_scale <= 1.0):
            raise ValueError(
                f"detection_scale must be in [0.1, 1.0], got {self.config.detection_scale}"
            )
        self._det_ksize = _normalize_odd_ksize(self.config.detection_blur_ksize)
        self._rec_ksize = _normalize_odd_ksize(self.config.recognition_blur_ksize)

    def process(self, frame: np.ndarray, roi: Optional[ROIConfig] = None) -> PreprocessedFrame:
        """Crop ROI, convert to grayscale, and build dual-resolution blurred streams."""
        t0 = time.perf_counter()
        frame_h, frame_w = frame.shape[:2]

        if self.config.use_roi and roi is not None:
            active_roi = roi.clamp(frame_w, frame_h)
            roi_bgr = active_roi.crop(frame)
        else:
            active_roi = ROIConfig(x=0, y=0, width=frame_w, height=frame_h)
            roi_bgr = frame

        if roi_bgr.ndim == 2:
            roi_gray_raw = roi_bgr
        else:
            roi_gray_raw = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        roi_h, roi_w = roi_gray_raw.shape[:2]

        # Full-resolution recognition stream (light blur keeps small pips crisp)
        if self._rec_ksize[0] > 1 or self._rec_ksize[1] > 1:
            recognition_gray = cv2.GaussianBlur(
                roi_gray_raw,
                self._rec_ksize,
                self.config.recognition_blur_sigma,
            )
        else:
            recognition_gray = roi_gray_raw

        # Downscaled detection stream (fast contour segmentation & noise suppression)
        det_w = max(1, int(round(roi_w * self.config.detection_scale)))
        det_h = max(1, int(round(roi_h * self.config.detection_scale)))
        if det_w != roi_w or det_h != roi_h:
            det_resized = cv2.resize(roi_gray_raw, (det_w, det_h), interpolation=cv2.INTER_AREA)
        else:
            det_resized = roi_gray_raw

        if self._det_ksize[0] > 1 or self._det_ksize[1] > 1:
            detection_gray = cv2.GaussianBlur(
                det_resized,
                self._det_ksize,
                self.config.detection_blur_sigma,
            )
        else:
            detection_gray = det_resized

        scale_x = det_w / float(roi_w)
        scale_y = det_h / float(roi_h)
        inv_scale_x = roi_w / float(det_w)
        inv_scale_y = roi_h / float(det_h)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return PreprocessedFrame(
            full_frame=frame,
            roi=active_roi,
            roi_bgr=roi_bgr,
            roi_gray_raw=roi_gray_raw,
            recognition_gray=recognition_gray,
            detection_gray=detection_gray,
            scale_x=scale_x,
            scale_y=scale_y,
            inv_scale_x=inv_scale_x,
            inv_scale_y=inv_scale_y,
            processing_time_ms=elapsed_ms,
        )
