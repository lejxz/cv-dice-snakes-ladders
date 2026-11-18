"""Unit tests and latency benchmark for Step 2: Dual-Resolution Preprocessing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config import CalibrationConfig, PreprocessingConfig, ROIConfig, load_calibration, save_calibration
from src.vision.preprocessing import FramePreprocessor


def test_dual_resolution_shapes_and_types() -> None:
    frame_1080p = np.zeros((1080, 1920, 3), dtype=np.uint8)
    roi = ROIConfig(x=160, y=90, width=1600, height=900)
    pre = FramePreprocessor(PreprocessingConfig(detection_scale=0.5))

    out = pre.process(frame_1080p, roi=roi)
    assert out.recognition_gray.shape == (900, 1600)
    assert out.detection_gray.shape == (450, 800)
    assert out.recognition_gray.dtype == np.uint8
    assert out.detection_gray.dtype == np.uint8
    assert abs(out.scale_x - 0.5) < 1e-6
    assert abs(out.inv_scale_x - 2.0) < 1e-6

    # Without ROI (full 1920x1080 -> 960x540 detection)
    pre_full = FramePreprocessor(PreprocessingConfig(use_roi=False, detection_scale=0.5))
    out_full = pre_full.process(frame_1080p)
    assert out_full.recognition_gray.shape == (1080, 1920)
    assert out_full.detection_gray.shape == (540, 960)


def test_noise_reduction_and_pip_edge_preservation() -> None:
    rng = np.random.default_rng(42)
    frame = np.full((720, 1280, 3), 28, dtype=np.uint8)
    # White die body with a 6 px dark pip
    cv2.rectangle(frame, (600, 320), (680, 400), (235, 235, 235), -1)
    cv2.circle(frame, (640, 360), 6, (20, 20, 20), -1)
    # Add sensor grain
    noise = rng.normal(0, 6.0, frame.shape)
    noisy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    roi = ROIConfig(x=240, y=100, width=800, height=520)
    pre = FramePreprocessor()
    out = pre.process(noisy, roi=roi)

    # Background noise std should drop significantly in detection_gray
    bg_raw_std = float(np.std(out.roi_gray_raw[10:110, 10:110]))
    bg_det_std = float(np.std(out.detection_gray[5:55, 5:55]))
    assert bg_det_std < bg_raw_std * 0.5

    # Pip center in recognition_gray should remain clearly dark compared to surrounding white face
    pip_val = int(out.recognition_gray[360 - 100, 640 - 240])
    face_val = int(out.recognition_gray[340 - 100, 620 - 240])
    assert face_val - pip_val > 150


def test_coordinate_and_rotated_rect_mapping() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    roi = ROIConfig(x=200, y=100, width=1000, height=600)
    pre = FramePreprocessor(PreprocessingConfig(detection_scale=0.5))
    out = pre.process(frame, roi=roi)

    rx, ry = out.detection_to_roi(120.0, 80.0)
    assert (rx, ry) == (240.0, 160.0)

    fx, fy = out.detection_to_full(120.0, 80.0)
    assert (fx, fy) == (440.0, 260.0)

    rect_det = ((120.0, 80.0), (40.0, 40.0), -15.0)
    rect_roi = out.scale_rect_to_roi(rect_det)
    assert rect_roi == ((240.0, 160.0), (80.0, 80.0), -15.0)

    rect_full = out.scale_rect_to_full(rect_det)
    assert rect_full == ((440.0, 260.0), (80.0, 80.0), -15.0)


def test_preprocessing_config_roundtrip(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cal.json"
    cfg = CalibrationConfig(
        preprocessing=PreprocessingConfig(
            use_roi=True,
            detection_scale=0.5,
            detection_blur_ksize=(5, 5),
            detection_blur_sigma=1.2,
            recognition_blur_ksize=(3, 3),
            recognition_blur_sigma=0.8,
        )
    )
    save_calibration(cfg, cfg_path)
    loaded = load_calibration(cfg_path)
    assert loaded == cfg


def test_preprocessing_latency_under_2ms() -> None:
    frame = np.full((1080, 1920, 3), 30, dtype=np.uint8)
    roi = ROIConfig(x=360, y=90, width=1200, height=900)
    pre = FramePreprocessor()

    # Warm-up
    for _ in range(10):
        pre.process(frame, roi=roi)

    times = [pre.process(frame, roi=roi).processing_time_ms for _ in range(200)]
    mean_ms = float(np.mean(times))
    assert mean_ms < 2.0, f"Mean preprocessing time {mean_ms:.3f} ms exceeded 2.0 ms target"
