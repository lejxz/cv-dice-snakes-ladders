"""Unit and integration tests for Step 1: Configuration, VideoStream, VideoRecorder, and Calibration CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from src.acquisition.camera import VideoStream
from src.acquisition.recorder import VideoRecorder
from src.config import AppConfig, ROIConfig, load_config, save_config


def test_roi_clamp_and_crop() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    roi = ROIConfig(x=-50, y=600, width=2000, height=500)
    clamped = roi.clamp(1280, 720)
    assert (clamped.x, clamped.y, clamped.width, clamped.height) == (0, 600, 1280, 120)
    assert roi.crop(frame).shape == (120, 1280, 3)


def test_config_save_and_load(tmp_path: Path) -> None:
    cfg_path = tmp_path / "calibration.json"
    original = AppConfig(
        source="synthetic",
        preferred_width=1920,
        preferred_height=1080,
        target_fps=30,
        roi=ROIConfig(x=100, y=80, width=640, height=480),
    )
    save_config(original, cfg_path)
    assert load_config(cfg_path) == original


def test_synthetic_video_stream_and_fps() -> None:
    cfg = AppConfig(source="synthetic", fallback_width=1280, fallback_height=720, target_fps=30)
    with VideoStream(cfg) as stream:
        assert (stream.width, stream.height) == (1280, 720)
        items = [stream.read() for _ in range(15)]

    assert all(item is not None for item in items)
    first_frame, _ = items[0]  # type: ignore[misc]
    _, last_fps = items[-1]    # type: ignore[misc]
    assert first_frame.shape == (720, 1280, 3)
    assert last_fps > 0.0


def test_recorder_and_video_file_playback(tmp_path: Path) -> None:
    rec = VideoRecorder(output_dir=tmp_path, fps=30.0)
    cfg = AppConfig(source="synthetic", fallback_width=1280, fallback_height=720, target_fps=30)

    with VideoStream(cfg) as stream:
        rec.start(stream.width, stream.height, prefix="test_clip")
        for _ in range(25):
            item = stream.read()
            assert item is not None
            frame, _ = item
            rec.write(frame)
        saved_video = rec.stop()

    assert saved_video is not None and saved_video.exists()
    assert rec.frame_count == 25

    replay_frames = 0
    with VideoStream(AppConfig(source=str(saved_video))) as replay_stream:
        assert (replay_stream.width, replay_stream.height) == (1280, 720)
        while replay_stream.read() is not None:
            replay_frames += 1

    assert replay_frames == 25


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
    assert len(list(tmp_path.glob("*.mp4"))) == 1
