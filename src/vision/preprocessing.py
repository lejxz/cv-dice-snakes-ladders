"""Dual-resolution preprocessing for fast die localization and crisp pip recognition."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import ROIConfig


@dataclass
class PreprocessedFrame:
    """Full-resolution and downscaled grayscale ROI views for one frame."""

    roi: ROIConfig
    roi_bgr: np.ndarray
    recognition_gray: np.ndarray  # Full-res ROI + light 3x3 blur (for pip reading)
    detection_gray: np.ndarray    # Downscaled ROI + 5x5 blur (for die detection)
    scale: float                  # Downscale factor (e.g. 0.5)
    processing_time_ms: float

    def to_roi_coords(self, x_det: float, y_det: float) -> Tuple[float, float]:
        """Convert (x, y) from downscaled detection_gray to full-res ROI coordinates."""
        return (x_det / self.scale, y_det / self.scale)

    def to_full_coords(self, x_det: float, y_det: float) -> Tuple[float, float]:
        """Convert (x, y) from downscaled detection_gray to full camera frame coordinates."""
        rx, ry = self.to_roi_coords(x_det, y_det)
        return (rx + self.roi.x, ry + self.roi.y)

    def scale_rect_to_roi(
        self, rect: Tuple[Tuple[float, float], Tuple[float, float], float]
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """Scale a cv2.minAreaRect ((cx, cy), (w, h), angle) from detection space to ROI space."""
        (cx, cy), (w, h), angle = rect
        return ((cx / self.scale, cy / self.scale), (w / self.scale, h / self.scale), angle)


def preprocess_frame(
    frame: np.ndarray,
    roi: Optional[ROIConfig] = None,
    scale: float = 0.5,
) -> PreprocessedFrame:
    """Crop to ROI, convert to grayscale, and build full-res + downscaled blurred streams."""
    if not (0.1 <= scale <= 1.0):
        raise ValueError(f"scale must be in [0.1, 1.0], got {scale}")

    t0 = time.perf_counter()
    h, w = frame.shape[:2]
    active_roi = roi.clamp(w, h) if roi is not None else ROIConfig(0, 0, w, h)
    roi_bgr = active_roi.crop(frame)

    gray = roi_bgr if roi_bgr.ndim == 2 else cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    recognition_gray = cv2.GaussianBlur(gray, (3, 3), 0.8)

    det_w = max(1, int(round(gray.shape[1] * scale)))
    det_h = max(1, int(round(gray.shape[0] * scale)))
    resized = cv2.resize(gray, (det_w, det_h), interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    detection_gray = cv2.GaussianBlur(resized, (5, 5), 1.2)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return PreprocessedFrame(
        roi=active_roi,
        roi_bgr=roi_bgr,
        recognition_gray=recognition_gray,
        detection_gray=detection_gray,
        scale=scale,
        processing_time_ms=elapsed_ms,
    )
