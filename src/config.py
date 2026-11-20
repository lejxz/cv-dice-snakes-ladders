"""Central configuration and calibration persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Union

import numpy as np


DEFAULT_CALIBRATION_PATH = Path("config/calibration.json")


@dataclass
class ROIConfig:
    """Rectangular Region of Interest (ROI) isolating the rolling play area."""

    x: int = 280
    y: int = 100
    width: int = 720
    height: int = 520

    def clamp(self, frame_w: int, frame_h: int) -> "ROIConfig":
        """Clamp ROI coordinates safely within the frame bounds."""
        if frame_w <= 0 or frame_h <= 0:
            raise ValueError(f"Frame dimensions must be positive, got ({frame_w}, {frame_h})")
        x = max(0, min(int(self.x), frame_w - 1))
        y = max(0, min(int(self.y), frame_h - 1))
        w = max(1, min(int(self.width), frame_w - x))
        h = max(1, min(int(self.height), frame_h - y))
        return ROIConfig(x=x, y=y, width=w, height=h)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Crop a frame to the clamped ROI."""
        h, w = frame.shape[:2]
        c = self.clamp(w, h)
        return frame[c.y : c.y + c.height, c.x : c.x + c.width]


@dataclass
class AppConfig:
    """Flat, readable configuration for acquisition, ROI, and preprocessing."""

    source: Union[int, str] = 0
    preferred_width: int = 1920
    preferred_height: int = 1080
    fallback_width: int = 1280
    fallback_height: int = 720
    target_fps: int = 30
    lock_camera: bool = True
    detection_scale: float = 0.5
    roi: ROIConfig = field(default_factory=ROIConfig)


def save_config(config: AppConfig, path: Union[str, Path] = DEFAULT_CALIBRATION_PATH) -> Path:
    """Save configuration to JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    return target


def load_config(path: Union[str, Path] = DEFAULT_CALIBRATION_PATH) -> AppConfig:
    """Load configuration from JSON, returning defaults if missing."""
    target = Path(path)
    if not target.exists():
        return AppConfig()
    with target.open("r", encoding="utf-8") as f:
        data = json.load(f)
    roi = ROIConfig(**data.pop("roi", {}))
    return AppConfig(roi=roi, **data)
