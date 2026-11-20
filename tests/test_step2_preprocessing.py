"""Unit tests and latency benchmark for Step 2: Dual-Resolution Preprocessing."""

from __future__ import annotations

import cv2
import numpy as np

from src.config import ROIConfig
from src.vision.preprocessing import preprocess_frame


def test_dual_resolution_shapes_and_types() -> None:
    frame_1080p = np.zeros((1080, 1920, 3), dtype=np.uint8)
    roi = ROIConfig(x=160, y=90, width=1600, height=900)

    out = preprocess_frame(frame_1080p, roi=roi, scale=0.5)
    assert out.recognition_gray.shape == (900, 1600)
    assert out.detection_gray.shape == (450, 800)
    assert out.recognition_gray.dtype == np.uint8
    assert out.detection_gray.dtype == np.uint8
    assert out.scale == 0.5

    # Without ROI (full 1920x1080 -> 960x540 detection)
    out_full = preprocess_frame(frame_1080p, roi=None, scale=0.5)
    assert out_full.recognition_gray.shape == (1080, 1920)
    assert out_full.detection_gray.shape == (540, 960)


def test_noise_reduction_and_pip_edge_preservation() -> None:
    rng = np.random.default_rng(42)
    frame = np.full((720, 1280, 3), 28, dtype=np.uint8)
    cv2.rectangle(frame, (600, 320), (680, 400), (235, 235, 235), -1)
    cv2.circle(frame, (640, 360), 6, (20, 20, 20), -1)
    noisy = np.clip(frame.astype(np.float32) + rng.normal(0, 6.0, frame.shape), 0, 255).astype(np.uint8)

    roi = ROIConfig(x=240, y=100, width=800, height=520)
    out = preprocess_frame(noisy, roi=roi, scale=0.5)

    # Background noise should drop in detection_gray relative to raw ROI
    raw_gray = cv2.cvtColor(out.roi_bgr, cv2.COLOR_BGR2GRAY)
    assert float(np.std(out.detection_gray[5:55, 5:55])) < float(np.std(raw_gray[10:110, 10:110])) * 0.5

    # Pip center in recognition_gray should remain clearly dark compared to surrounding white face
    pip_val = int(out.recognition_gray[360 - 100, 640 - 240])
    face_val = int(out.recognition_gray[340 - 100, 620 - 240])
    assert face_val - pip_val > 150


def test_coordinate_and_rotated_rect_mapping() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    roi = ROIConfig(x=200, y=100, width=1000, height=600)
    out = preprocess_frame(frame, roi=roi, scale=0.5)

    assert out.to_roi_coords(120.0, 80.0) == (240.0, 160.0)
    assert out.to_full_coords(120.0, 80.0) == (440.0, 260.0)

    rect_det = ((120.0, 80.0), (40.0, 40.0), -15.0)
    assert out.scale_rect_to_roi(rect_det) == ((240.0, 160.0), (80.0, 80.0), -15.0)


def test_preprocessing_latency_under_2ms() -> None:
    frame = np.full((1080, 1920, 3), 30, dtype=np.uint8)
    roi = ROIConfig(x=360, y=90, width=1200, height=900)

    for _ in range(10):
        preprocess_frame(frame, roi=roi, scale=0.5)

    times = [preprocess_frame(frame, roi=roi, scale=0.5).processing_time_ms for _ in range(200)]
    mean_ms = float(np.mean(times))
    assert mean_ms < 2.0, f"Mean preprocessing time {mean_ms:.3f} ms exceeded 2.0 ms target"
