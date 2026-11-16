"""Raw video clip and timestamp recorder for offline evaluation and roll-settle tuning."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO, Tuple, Union

import cv2
import numpy as np


class VideoRecorder:
    """Records unannotated raw video frames and frame-level timestamps."""

    def __init__(self, output_dir: Union[str, Path] = "data/recordings", fps: float = 30.0) -> None:
        self.output_dir = Path(output_dir)
        self.fps = float(fps) if fps > 0 else 30.0
        self._writer: Optional[cv2.VideoWriter] = None
        self._csv_file: Optional[TextIO] = None
        self._csv_writer = None
        self._video_path: Optional[Path] = None
        self._csv_path: Optional[Path] = None
        self._frame_count = 0
        self._frame_size: Optional[Tuple[int, int]] = None

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    @property
    def video_path(self) -> Optional[Path]:
        return self._video_path

    @property
    def csv_path(self) -> Optional[Path]:
        return self._csv_path

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start(self, frame_width: int, frame_height: int, prefix: str = "roll") -> Path:
        """Start recording to a new timestamped .mp4 file and sidecar _timestamps.csv."""
        if self.is_recording:
            return self._video_path  # type: ignore[return-value]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._video_path = self.output_dir / f"{prefix}_{stamp}.mp4"
        self._csv_path = self.output_dir / f"{prefix}_{stamp}_timestamps.csv"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._frame_size = (int(frame_width), int(frame_height))
        self._writer = cv2.VideoWriter(str(self._video_path), fourcc, self.fps, self._frame_size)
        if not self._writer.isOpened():
            self._writer = None
            raise RuntimeError(f"Failed to open VideoWriter at {self._video_path}")

        self._csv_file = self._csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["frame_idx", "timestamp_sec", "fps"])
        self._frame_count = 0
        return self._video_path

    def write(self, frame: np.ndarray, timestamp: float, fps: float) -> None:
        """Write a single raw frame and its timing metadata."""
        if not self.is_recording or self._writer is None:
            return

        h, w = frame.shape[:2]
        if self._frame_size is not None and (w, h) != self._frame_size:
            frame = cv2.resize(frame, self._frame_size)

        self._writer.write(frame)
        if self._csv_writer is not None:
            self._csv_writer.writerow([self._frame_count, f"{timestamp:.6f}", f"{fps:.2f}"])
        self._frame_count += 1

    def stop(self) -> Optional[Path]:
        """Finalize and close the video and timestamp files."""
        saved_path = self._video_path
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        return saved_path

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
