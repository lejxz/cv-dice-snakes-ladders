"""Unit and integration tests for Step 1: Configuration, VideoStream, VideoRecorder, and Calibration CLI."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.acquisition.camera import FPSMeter, VideoStream
from src.acquisition.recorder import VideoRecorder
from src.config import CalibrationConfig, CameraConfig, ROIConfig, load_calibration, save_calibration


def test_roi_clamp_and_crop() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    roi = ROIConfig(x=-50, y=600, width=2000, height=500)
    clamped = roi.clamp(1280, 720)
    assert clamped.x == 0
    assert clamped.y == 600
    assert clamped.width == 1280
    assert clamped.height == 120

    cropped = roi.crop(frame)
    assert cropped.shape == (120, 1280, 3)


def test_config_save_and_load(tmp_path: Path) -> None:
    cfg_path = tmp_path / "calibration.json"
    original = CalibrationConfig(
        camera=CameraConfig(source="synthetic", preferred_width=1920, preferred_height=1080, target_fps=30),
        roi=ROIConfig(x=100, y=80, width=640, height=480),
    )
    save_calibration(original, cfg_path)
    loaded = load_calibration(cfg_path)
    assert loaded == original


def test_synthetic_video_stream_and_fps() -> None:
    cfg = CameraConfig(source="synthetic", fallback_width=1280, fallback_height=720, target_fps=30)
    with VideoStream(cfg) as stream:
        assert stream.is_opened
        assert stream.resolution == (1280, 720)
        packets = [stream.read() for _ in range(15)]

    assert all(p is not None for p in packets)
    assert packets[0].frame.shape == (720, 1280, 3)  # type: ignore[union-attr]
    assert packets[-1].fps > 0.0  # type: ignore[union-attr]


def test_fps_meter() -> None:
    meter = FPSMeter(window_size=10)
    assert meter.fps == 0.0
    for i in range(5):
        meter.tick(now=float(i) * 0.05)  # 20 FPS
    assert abs(meter.fps - 20.0) < 1e-6


def test_recorder_and_video_file_playback(tmp_path: Path) -> None:
    rec = VideoRecorder(output_dir=tmp_path, fps=30.0)
    cfg = CameraConfig(source="synthetic", fallback_width=1280, fallback_height=720, target_fps=30)

    with VideoStream(cfg) as stream:
        w, h = stream.resolution
        rec.start(w, h, prefix="test_clip")
        for _ in range(25):
            pkt = stream.read()
            assert pkt is not None
            rec.write(pkt.frame, pkt.timestamp, pkt.fps)
        saved_video = rec.stop()

    assert saved_video is not None and saved_video.exists()
    assert rec.csv_path is not None and rec.csv_path.exists()
    assert rec.frame_count == 25

    with rec.csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 26  # header + 25 frames

    # Replay saved mp4 through VideoStream
    replay_cfg = CameraConfig(source=str(saved_video))
    replayed_frames = 0
    with VideoStream(replay_cfg) as replay_stream:
        assert replay_stream.resolution == (1280, 720)
        while replay_stream.read() is not None:
            replayed_frames += 1

    assert replayed_frames == 25


def test_calibrate_headless_cli(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.acquisition.calibrate",
        "--source",
        "synthetic",
        "--headless",
        "--frames",
        "30",
        "--record",
        "--output-dir",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    assert "Processed 30 frames" in result.stdout
    mp4_files = list(tmp_path.glob("*.mp4"))
    assert len(mp4_files) == 1
