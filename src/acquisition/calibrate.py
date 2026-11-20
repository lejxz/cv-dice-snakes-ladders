"""Play-area ROI calibration, camera lock verification, and raw clip recording CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.acquisition.camera import VideoStream
from src.acquisition.recorder import VideoRecorder
from src.config import DEFAULT_CALIBRATION_PATH, ROIConfig, load_config, save_config


class _ROISelector:
    """Tracks mouse-drag ROI selection in the OpenCV preview window."""

    def __init__(self, initial_roi: ROIConfig) -> None:
        self.roi = initial_roi
        self.dragging = False
        self.start_pt = (0, 0)
        self.end_pt = (0, 0)

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start_pt = self.end_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            x0, y0 = self.start_pt
            x1, y1 = x, y
            rw, rh = abs(x1 - x0), abs(y1 - y0)
            if rw >= 40 and rh >= 40:
                self.roi = ROIConfig(x=min(x0, x1), y=min(y0, y1), width=rw, height=rh)


def draw_hud(
    frame: np.ndarray,
    roi: ROIConfig,
    fps: float,
    is_recording: bool,
    lock_status: dict,
    selector: Optional[_ROISelector] = None,
) -> np.ndarray:
    """Overlay ROI box, FPS, camera lock state, and key controls on a frame copy."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    c = roi.clamp(w, h)

    cv2.rectangle(vis, (c.x, c.y), (c.x + c.width, c.y + c.height), (0, 220, 120), 2)
    if selector is not None and selector.dragging:
        cv2.rectangle(vis, selector.start_pt, selector.end_pt, (0, 200, 255), 1)

    rec_text = "REC [ON]" if is_recording else "REC [OFF]"
    rec_color = (0, 0, 255) if is_recording else (200, 200, 200)
    cv2.putText(
        vis,
        f"Res: {w}x{h} | FPS: {fps:5.1f} | ROI: ({c.x},{c.y},{c.width}x{c.height})",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"{rec_text} | Locks: exp={lock_status.get('auto_exposure')} foc={lock_status.get('autofocus')} wb={lock_status.get('auto_wb')}",
        (16, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        rec_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        "Drag Mouse: Set ROI | [R] Record | [L] Lock Cam | [S] Save | [Q] Quit",
        (16, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return vis


def run_calibration(
    source: Optional[str] = None,
    config_path: Path = DEFAULT_CALIBRATION_PATH,
    headless: bool = False,
    max_frames: int = 60,
    record: bool = False,
    output_dir: Path = Path("data/recordings"),
) -> int:
    cfg = load_config(config_path)
    if source is not None:
        cfg.source = int(source) if source.isdigit() else source

    selector = _ROISelector(cfg.roi)
    recorder = VideoRecorder(output_dir=output_dir, fps=cfg.target_fps)
    window_name = "Step 1: Acquisition & Play-Area Calibration"

    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, selector.on_mouse)

    processed = 0
    last_fps = 0.0
    try:
        with VideoStream(cfg) as stream:
            selector.roi = selector.roi.clamp(stream.width, stream.height)
            print(f"[Acquisition] Opened source={cfg.source} | Resolution={stream.width}x{stream.height}")
            print(f"[Acquisition] Camera lock status: {stream.lock_status}")

            if record:
                rec_path = recorder.start(stream.width, stream.height)
                print(f"[Recorder] Recording raw frames to {rec_path}")

            while True:
                item = stream.read()
                if item is None:
                    break
                frame, last_fps = item
                processed += 1

                if recorder.is_recording:
                    recorder.write(frame)

                if headless:
                    if processed >= max_frames:
                        break
                    continue

                hud = draw_hud(frame, selector.roi, last_fps, recorder.is_recording, stream.lock_status, selector)
                cv2.imshow(window_name, hud)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("s"):
                    cfg.roi = selector.roi.clamp(stream.width, stream.height)
                    saved = save_config(cfg, config_path)
                    print(f"[Calibration] Saved config to {saved}")
                elif key == ord("l"):
                    print(f"[Camera] Re-applied control locks: {stream.lock_camera_controls()}")
                elif key == ord("r"):
                    if recorder.is_recording:
                        print(f"[Recorder] Stopped -> {recorder.stop()}")
                    else:
                        print(f"[Recorder] Started -> {recorder.start(stream.width, stream.height)}")
    finally:
        if recorder.is_recording:
            print(f"[Recorder] Finalized -> {recorder.stop()} ({recorder.frame_count} frames)")
        if not headless:
            cv2.destroyAllWindows()

    print(f"[Acquisition] Processed {processed} frames | Measured FPS: {last_fps:.1f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 1: Video Acquisition & Play-Area ROI Calibration")
    parser.add_argument("--source", type=str, default=None, help="Camera index (e.g. 0), video path, or 'synthetic'")
    parser.add_argument("--config", type=Path, default=DEFAULT_CALIBRATION_PATH, help="Calibration JSON path")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--frames", type=int, default=60, help="Frames to process in headless mode")
    parser.add_argument("--record", action="store_true", help="Record raw frames to .mp4")
    parser.add_argument("--output-dir", type=Path, default=Path("data/recordings"), help="Directory for recordings")
    args = parser.parse_args()
    return run_calibration(
        source=args.source,
        config_path=args.config,
        headless=args.headless,
        max_frames=args.frames,
        record=args.record,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
