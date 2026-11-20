"""Video acquisition wrapper for live webcams, recorded clips, and synthetic test frames."""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

from src.config import AppConfig


def make_synthetic_frame(frame_idx: int, width: int = 1280, height: int = 720, fps: int = 30) -> np.ndarray:
    """Create a synthetic dark-surface frame with a settling white 16 mm die."""
    frame = np.full((height, width, 3), 28, dtype=np.uint8)
    t = frame_idx / float(max(1, fps))
    cx = int(width * 0.5 + math.sin(t * 2.0) * 18.0 * math.exp(-t * 0.8))
    cy = int(height * 0.5 + math.cos(t * 2.5) * 12.0 * math.exp(-t * 0.8))
    half = 38  # ~76x76 px die at 720p

    cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half), (238, 238, 238), -1)
    for dx, dy in [(-18, -18), (18, -18), (0, 0), (-18, 18), (18, 18)]:
        cv2.circle(frame, (cx + dx, cy + dy), 6, (25, 25, 25), -1)
    return frame


class VideoStream:
    """Unified reader for a USB/phone webcam, recorded video file, or 'synthetic' source."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.width: int = 0
        self.height: int = 0
        self.fps: float = 0.0
        self.frame_idx: int = 0
        self.lock_status: Dict[str, Optional[float]] = {}

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_synthetic: bool = False
        self._is_file: bool = False
        self._timestamps: deque[float] = deque(maxlen=30)

    def open(self) -> "VideoStream":
        src: Union[int, str] = self.config.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        if isinstance(src, str) and src.lower() == "synthetic":
            self._is_synthetic = True
            self.width = self.config.fallback_width
            self.height = self.config.fallback_height
            self.lock_status = {"auto_exposure": 0.0, "autofocus": 0.0, "auto_wb": 0.0}
            return self

        if isinstance(src, str):
            if not Path(src).exists():
                raise FileNotFoundError(f"Video file not found: {src}")
            self._is_file = True
            self._cap = cv2.VideoCapture(src)
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {src}")
            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return self

        # Live camera device
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera device {src}")

        self._cap.set(cv2.CAP_PROP_FPS, float(self.config.target_fps))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.config.preferred_width))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.config.preferred_height))
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if w < self.config.fallback_width or h < self.config.fallback_height:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.config.fallback_width))
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.config.fallback_height))
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.width, self.height = w, h
        if self.config.lock_camera:
            self.lock_camera_controls()
        return self

    def lock_camera_controls(self) -> Dict[str, Optional[float]]:
        """Lock auto-exposure, autofocus, and auto white balance on live cameras."""
        if self._cap is None or self._is_file or self._is_synthetic:
            return self.lock_status

        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode (V4L2)
        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self._cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.lock_status = {
            "auto_exposure": self._cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "autofocus": self._cap.get(cv2.CAP_PROP_AUTOFOCUS),
            "auto_wb": self._cap.get(cv2.CAP_PROP_AUTO_WB),
        }
        return self.lock_status

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        """Return (frame, measured_fps), or None on EOF / capture failure."""
        if self._is_synthetic:
            frame = make_synthetic_frame(self.frame_idx, self.width, self.height, self.config.target_fps)
        elif self._cap is not None and self._cap.isOpened():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return None
        else:
            raise RuntimeError("VideoStream is not open. Call open() before read().")

        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) >= 2:
            dt = self._timestamps[-1] - self._timestamps[0]
            self.fps = (len(self._timestamps) - 1) / dt if dt > 0 else 0.0

        self.frame_idx += 1
        return frame, self.fps

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoStream":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
