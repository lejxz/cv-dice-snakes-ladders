"""Raw .mp4 video clip recorder for offline evaluation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


class VideoRecorder:
    """Records raw video frames to an .mp4 file in data/recordings/."""

    def __init__(self, output_dir: Union[str, Path] = "data/recordings", fps: float = 30.0) -> None:
        self.output_dir = Path(output_dir)
        self.fps = float(fps) if fps > 0 else 30.0
        self.video_path: Optional[Path] = None
        self.frame_count: int = 0
        self._writer: Optional[cv2.VideoWriter] = None

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    def start(self, width: int, height: int, prefix: str = "roll") -> Path:
        if self._writer is not None:
            return self.video_path  # type: ignore[return-value]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = self.output_dir / f"{prefix}_{stamp}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.video_path), fourcc, self.fps, (int(width), int(height)))
        if not self._writer.isOpened():
            self._writer = None
            raise RuntimeError(f"Failed to open VideoWriter at {self.video_path}")

        self.frame_count = 0
        return self.video_path

    def write(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)
            self.frame_count += 1

    def stop(self) -> Optional[Path]:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        return self.video_path

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
