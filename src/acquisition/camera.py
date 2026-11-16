"""Video acquisition wrapper supporting USB/smartphone webcams, recorded clips, and synthetic streams."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

from src.config import CameraConfig


@dataclass
class FramePacket:
    """Container for a single acquired video frame and its metadata."""

    frame: np.ndarray
    frame_idx: int
    timestamp: float
    fps: float


class FPSMeter:
    """Rolling-window FPS estimator."""

    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=max(2, window_size))

    def tick(self, now: Optional[float] = None) -> float:
        t = time.perf_counter() if now is None else now
        self._timestamps.append(t)
        return self.fps

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0.0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


class _SyntheticDiceGenerator:
    """Generates synthetic 1280x720 dark-surface frames with a white 16 mm die for headless testing."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_idx = 0

    def read(self) -> Tuple[bool, np.ndarray]:
        # Dark matte surface (intensity ~28) with subtle sensor grain
        frame = np.full((self.height, self.width, 3), 28, dtype=np.uint8)

        # Simulate a die settling in the center of the play area
        t = self.frame_idx / float(max(1, self.fps))
        cx = int(self.width * 0.5 + math.sin(t * 2.0) * 18.0 * math.exp(-t * 0.8))
        cy = int(self.height * 0.5 + math.cos(t * 2.5) * 12.0 * math.exp(-t * 0.8))
        half_size = 38  # ~76x76 px die at 720p

        # White die body
        cv2.rectangle(
            frame,
            (cx - half_size, cy - half_size),
            (cx + half_size, cy + half_size),
            (238, 238, 238),
            thickness=-1,
        )
        # Draw 5 pips (four corners + center)
        offset = 18
        for dx, dy in [(-offset, -offset), (offset, -offset), (0, 0), (-offset, offset), (offset, offset)]:
            cv2.circle(frame, (cx + dx, cy + dy), 6, (25, 25, 25), thickness=-1)

        self.frame_idx += 1
        return True, frame


class VideoStream:
    """Unified video source for live USB/phone cameras, video files, and synthetic test streams."""

    def __init__(self, config: Optional[CameraConfig] = None, pace_video_file: bool = False) -> None:
        self.config = config or CameraConfig()
        self.pace_video_file = pace_video_file
        self._cap: Optional[cv2.VideoCapture] = None
        self._synthetic: Optional[_SyntheticDiceGenerator] = None
        self._fps_meter = FPSMeter(window_size=30)
        self._frame_idx = 0
        self._is_file = False
        self._is_synthetic = False
        self._negotiated_resolution: Tuple[int, int] = (0, 0)
        self._source_fps: float = float(self.config.target_fps)
        self._last_read_time: Optional[float] = None
        self._lock_status: Dict[str, Optional[float]] = {}

    @property
    def is_opened(self) -> bool:
        if self._is_synthetic:
            return self._synthetic is not None
        return self._cap is not None and self._cap.isOpened()

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._negotiated_resolution

    @property
    def lock_status(self) -> Dict[str, Optional[float]]:
        return dict(self._lock_status)

    @property
    def source_fps(self) -> float:
        return self._source_fps

    def open(self) -> "VideoStream":
        """Open the configured video source and negotiate resolution/camera locks."""
        src: Union[int, str] = self.config.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        if isinstance(src, str) and src.lower() == "synthetic":
            self._is_synthetic = True
            self._synthetic = _SyntheticDiceGenerator(
                width=self.config.fallback_width,
                height=self.config.fallback_height,
                fps=self.config.target_fps,
            )
            self._negotiated_resolution = (self.config.fallback_width, self.config.fallback_height)
            self._source_fps = float(self.config.target_fps)
            self._lock_status = {"auto_exposure": 0.0, "autofocus": 0.0, "auto_wb": 0.0}
            return self

        if isinstance(src, str):
            path = Path(src)
            if not path.exists():
                raise FileNotFoundError(f"Video file not found: {src}")
            self._is_file = True
            self._cap = cv2.VideoCapture(str(path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {src}")
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(self._cap.get(cv2.CAP_PROP_FPS))
            self._negotiated_resolution = (w, h)
            self._source_fps = fps if fps > 0 else float(self.config.target_fps)
            return self

        # Live camera device index
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera device index {src}")

        self._negotiated_resolution = self._negotiate_resolution()
        reported_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._source_fps = reported_fps if reported_fps > 0 else float(self.config.target_fps)
        self.lock_camera_controls()
        return self

    def _negotiate_resolution(self) -> Tuple[int, int]:
        """Try preferred 1080p resolution, falling back to 720p if unsupported."""
        assert self._cap is not None
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

        return (w, h)

    def lock_camera_controls(self) -> Dict[str, Optional[float]]:
        """Attempt to lock exposure, focus, and white balance on a live camera."""
        if self._cap is None or self._is_file or self._is_synthetic:
            return self._lock_status

        if self.config.lock_exposure:
            # V4L2 uses 1 for manual exposure; DirectShow uses 0.25. Try 1 first.
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            if self.config.exposure_value is not None:
                self._cap.set(cv2.CAP_PROP_EXPOSURE, float(self.config.exposure_value))

        if self.config.lock_focus:
            self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            if self.config.focus_value is not None:
                self._cap.set(cv2.CAP_PROP_FOCUS, float(self.config.focus_value))

        if self.config.lock_white_balance:
            self._cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            if self.config.wb_temperature is not None:
                self._cap.set(cv2.CAP_PROP_WB_TEMPERATURE, float(self.config.wb_temperature))

        self._lock_status = {
            "auto_exposure": self._cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "autofocus": self._cap.get(cv2.CAP_PROP_AUTOFOCUS),
            "auto_wb": self._cap.get(cv2.CAP_PROP_AUTO_WB),
        }
        return dict(self._lock_status)

    def read(self) -> Optional[FramePacket]:
        """Read the next frame packet, or return None on EOF / capture failure."""
        if not self.is_opened:
            raise RuntimeError("VideoStream is not open. Call open() before read().")

        if self._is_file and self.pace_video_file and self._last_read_time is not None:
            target_dt = 1.0 / max(1.0, self._source_fps)
            elapsed = time.perf_counter() - self._last_read_time
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

        if self._is_synthetic:
            assert self._synthetic is not None
            ok, frame = self._synthetic.read()
        else:
            assert self._cap is not None
            ok, frame = self._cap.read()

        if not ok or frame is None:
            return None

        now = time.perf_counter()
        self._last_read_time = now
        fps = self._fps_meter.tick(now)
        packet = FramePacket(
            frame=frame,
            frame_idx=self._frame_idx,
            timestamp=now,
            fps=fps,
        )
        self._frame_idx += 1
        return packet

    def release(self) -> None:
        """Release underlying video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._synthetic = None

    def __enter__(self) -> "VideoStream":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
