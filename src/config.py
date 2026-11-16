"""Central configuration and calibration persistence for the CV Snakes & Ladders pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


DEFAULT_CALIBRATION_PATH = Path("config/calibration.json")


@dataclass
class ROIConfig:
    """Rectangular Region of Interest (ROI) isolating the rolling play area."""

    x: int = 280
    y: int = 100
    width: int = 720
    height: int = 520

    def clamp(self, frame_width: int, frame_height: int) -> "ROIConfig":
        """Return a new ROIConfig clamped safely within [0, frame_width) x [0, frame_height)."""
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(f"Frame dimensions must be positive, got ({frame_width}, {frame_height})")

        x = max(0, min(int(self.x), frame_width - 1))
        y = max(0, min(int(self.y), frame_height - 1))
        w = max(1, min(int(self.width), frame_width - x))
        h = max(1, min(int(self.height), frame_height - y))
        return ROIConfig(x=x, y=y, width=w, height=h)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Crop a frame to the clamped ROI."""
        h, w = frame.shape[:2]
        clamped = self.clamp(w, h)
        return frame[clamped.y : clamped.y + clamped.height, clamped.x : clamped.x + clamped.width]


@dataclass
class CameraConfig:
    """Camera acquisition and driver-lock configuration."""

    source: Union[int, str] = 0
    preferred_width: int = 1920
    preferred_height: int = 1080
    fallback_width: int = 1280
    fallback_height: int = 720
    target_fps: int = 30
    lock_exposure: bool = True
    lock_focus: bool = True
    lock_white_balance: bool = True
    exposure_value: Optional[float] = None
    focus_value: Optional[float] = None
    wb_temperature: Optional[float] = None


@dataclass
class CalibrationConfig:
    """Top-level persisted calibration settings."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationConfig":
        cam_data = data.get("camera", {})
        roi_data = data.get("roi", {})
        return cls(
            camera=CameraConfig(**cam_data),
            roi=ROIConfig(**roi_data),
        )


def save_calibration(config: CalibrationConfig, path: Union[str, Path] = DEFAULT_CALIBRATION_PATH) -> Path:
    """Serialize calibration configuration to JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
    return target


def load_calibration(path: Union[str, Path] = DEFAULT_CALIBRATION_PATH) -> CalibrationConfig:
    """Load calibration configuration from JSON, returning defaults if the file does not exist."""
    target = Path(path)
    if not target.exists():
        return CalibrationConfig()
    with target.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return CalibrationConfig.from_dict(data)
