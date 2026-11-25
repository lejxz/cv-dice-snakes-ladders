"""Raw .mp4 video clip recorder for offline evaluation."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

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


class GroundTruthLogger:
    """Appends and manages ground-truth roll labels in a CSV file."""

    FIELDNAMES = ["video_file", "roll_index", "ground_truth", "frame_idx", "timestamp"]

    def __init__(self, csv_path: Union[str, Path] = "data/ground_truth.csv") -> None:
        self.csv_path = Path(csv_path)
        self.total_count: int = self._count_existing_rows()

    def _count_existing_rows(self) -> int:
        if not self.csv_path.exists():
            return 0
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return sum(1 for _ in reader)

    def log_roll(self, video_path: Path, value: int, frame_idx: int, roll_index: int) -> None:
        """Append a single ground-truth roll entry (1-6) and flush immediately."""
        if not 1 <= value <= 6:
            raise ValueError(f"Roll value must be between 1 and 6, got {value}")

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "video_file": video_path.name,
                    "roll_index": roll_index,
                    "ground_truth": value,
                    "frame_idx": frame_idx,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )
            f.flush()
        self.total_count += 1

    def undo_last(self, video_path: Optional[Path] = None) -> Optional[int]:
        """Remove the last logged row (optionally restricted to video_path) and return its value."""
        if not self.csv_path.exists():
            return None

        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            rows: List[dict] = list(csv.DictReader(f))

        if not rows:
            return None

        if video_path is not None and rows[-1]["video_file"] != video_path.name:
            return None

        removed = rows.pop()
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()

        self.total_count = max(0, self.total_count - 1)
        return int(removed["ground_truth"])
