"""Unit tests for Step 3: Classical Die Detection & Upright Warping."""

from __future__ import annotations

import cv2
import numpy as np

from src.config import ROIConfig
from src.vision.die_detector import detect_die
from src.vision.preprocessing import preprocess_frame


def _draw_rotated_die(
    frame: np.ndarray,
    center: tuple[int, int],
    size: int = 80,
    angle_deg: float = 0.0,
) -> None:
    """Draw a white square die with a 5-pip pattern rotated by angle_deg."""
    cx, cy = center
    half = size / 2.0
    corners = np.array(
        [[-half, -half], [half, -half], [half, half], [-half, half]],
        dtype=np.float32,
    )
    rad = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]], dtype=np.float32)
    pts = np.round(corners @ rot.T + np.array([cx, cy], dtype=np.float32)).astype(np.int32)
    cv2.fillPoly(frame, [pts], (238, 238, 238))

    pip_off = size * 0.24
    for px, py in [(-pip_off, -pip_off), (pip_off, -pip_off), (0.0, 0.0), (-pip_off, pip_off), (pip_off, pip_off)]:
        p = np.array([px, py], dtype=np.float32) @ rot.T + np.array([cx, cy], dtype=np.float32)
        cv2.circle(frame, (int(round(p[0])), int(round(p[1]))), 6, (25, 25, 25), -1)


def test_detect_die_at_arbitrary_rotations() -> None:
    roi = ROIConfig(x=200, y=100, width=800, height=500)
    for angle in (0.0, 25.0, 45.0, 130.0):
        frame = np.full((720, 1280, 3), 28, dtype=np.uint8)
        _draw_rotated_die(frame, center=(600, 350), size=84, angle_deg=angle)

        prep = preprocess_frame(frame, roi=roi, scale=0.5)
        det = detect_die(prep)

        assert det is not None, f"Failed to detect die at angle={angle}"
        assert det.face_gray.shape == (120, 120)
        assert abs(det.center_roi[0] - 400.0) < 3.0
        assert abs(det.center_roi[1] - 250.0) < 3.0
        # Upright face should contain bright white background and dark pip pixels
        assert int(det.face_gray.max()) > 200
        assert int(det.face_gray.min()) < 60


def test_empty_tray_returns_none() -> None:
    frame = np.full((720, 1280, 3), 28, dtype=np.uint8)
    # Add tiny bright dust specks that morphological opening should reject
    cv2.circle(frame, (400, 300), 2, (230, 230, 230), -1)
    cv2.circle(frame, (700, 420), 1, (240, 240, 240), -1)

    prep = preprocess_frame(frame, roi=ROIConfig(200, 100, 800, 500), scale=0.5)
    assert detect_die(prep) is None


def test_multiple_candidates_rejected() -> None:
    frame = np.full((720, 1280, 3), 28, dtype=np.uint8)
    _draw_rotated_die(frame, center=(450, 320), size=80, angle_deg=10.0)
    _draw_rotated_die(frame, center=(750, 360), size=80, angle_deg=35.0)

    prep = preprocess_frame(frame, roi=ROIConfig(200, 100, 800, 500), scale=0.5)
    assert detect_die(prep) is None


def test_non_square_distractor_rejected() -> None:
    frame = np.full((720, 1280, 3), 28, dtype=np.uint8)
    # Elongated bright bar (aspect ratio ~0.175 < 0.75)
    cv2.rectangle(frame, (450, 300), (650, 335), (235, 235, 235), -1)

    prep = preprocess_frame(frame, roi=ROIConfig(200, 100, 800, 500), scale=0.5)
    assert detect_die(prep) is None


def test_adaptive_threshold_mode() -> None:
    frame = np.full((720, 1280, 3), 20, dtype=np.uint8)
    # Add horizontal lighting gradient across the tray
    gradient = np.linspace(0, 70, 1280, dtype=np.uint8)[None, :, None]
    frame = np.clip(frame.astype(np.int16) + gradient, 0, 255).astype(np.uint8)
    _draw_rotated_die(frame, center=(600, 350), size=84, angle_deg=20.0)

    prep = preprocess_frame(frame, roi=ROIConfig(200, 100, 800, 500), scale=0.5)
    det = detect_die(prep, use_adaptive=True)
    assert det is not None
    assert det.face_gray.shape == (120, 120)
