"""Play-area ROI calibration, camera lock verification, and raw clip recording CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.acquisition.camera import VideoStream
from src.acquisition.recorder import VideoRecorder
from src.config import DEFAULT_CALIBRATION_PATH, ROIConfig, load_calibration, save_calibration


class _ROISelectorState:
    """Tracks interactive mouse-drag ROI selection in GUI mode."""

    def __init__(self, initial_roi: ROIConfig) -> None:
        self.roi = initial_roi
        self.dragging = False
        self.start_pt = (0, 0)
        self.end_pt = (0, 0)

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start_pt = (x, y)
            self.end_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.end_pt = (x, y)
            x0, y0 = self.start_pt
            x1, y1 = self.end_pt
            rx, ry = min(x0, x1), min(y0, y1)
            rw, rh = abs(x1 - x0), abs(y1 - y0)
            if rw >= 40 and rh >= 40:
                self.roi = ROIConfig(x=rx, y=ry, width=rw, height=rh)


def draw_calibration_hud(
    frame: np.ndarray,
    roi: ROIConfig,
    fps: float,
    is_recording: bool,
    lock_status: dict,
    drag_state: Optional[_ROISelectorState] = None,
) -> np.ndarray:
    """Render ROI box, FPS, lock status, and key hints onto a copy of the frame."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    clamped = roi.clamp(w, h)

    # Draw active ROI rectangle
    cv2.rectangle(
        vis,
        (clamped.x, clamped.y),
        (clamped.x + clamped.width, clamped.y + clamped.height),
        (0, 220, 120),
        2,
    )

    # Draw in-progress drag box if user is dragging
    if drag_state is not None and drag_state.dragging:
        cv2.rectangle(vis, drag_state.start_pt, drag_state.end_pt, (0, 200, 255), 1)

    rec_label = "REC [ON]" if is_recording else "REC [OFF]"
    rec_color = (0, 0, 255) if is_recording else (200, 200, 200)

    cv2.putText(
        vis,
        f"Res: {w}x{h} | FPS: {fps:5.1f} | ROI: ({clamped.x},{clamped.y},{clamped.width}x{clamped.height})",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"{rec_label} | Locks: exp={lock_status.get('auto_exposure')} foc={lock_status.get('autofocus')} wb={lock_status.get('auto_wb')}",
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
    """Run the acquisition & calibration loop in GUI or headless mode."""
    cal_cfg = load_calibration(config_path)
    if source is not None:
        cal_cfg.camera.source = int(source) if source.isdigit() else source

    selector = _ROISelectorState(cal_cfg.roi)
    recorder = VideoRecorder(output_dir=output_dir, fps=cal_cfg.camera.target_fps)

    window_name = "Step 1: Acquisition & Play-Area Calibration"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, selector.on_mouse)

    processed = 0
    last_fps = 0.0
    try:
        with VideoStream(cal_cfg.camera) as stream:
            w, h = stream.resolution
            selector.roi = selector.roi.clamp(w, h)
            print(f"[Acquisition] Opened source={cal_cfg.camera.source} | Resolution={w}x{h} | TargetFPS={cal_cfg.camera.target_fps}")
            print(f"[Acquisition] Camera lock status: {stream.lock_status}")

            if record:
                rec_path = recorder.start(w, h)
                print(f"[Recorder] Recording raw frames to {rec_path}")

            while True:
                packet = stream.read()
                if packet is None:
                    break

                processed += 1
                last_fps = packet.fps

                if recorder.is_recording:
                    recorder.write(packet.frame, packet.timestamp, packet.fps)

                if headless:
                    if processed >= max_frames:
                        break
                    continue

                hud = draw_calibration_hud(
                    packet.frame,
                    selector.roi,
                    packet.fps,
                    recorder.is_recording,
                    stream.lock_status,
                    selector,
                )
                cv2.imshow(window_name, hud)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("s"):
                    cal_cfg.roi = selector.roi.clamp(w, h)
                    saved = save_calibration(cal_cfg, config_path)
                    print(f"[Calibration] Saved ROI & camera config to {saved}")
                elif key == ord("l"):
                    status = stream.lock_camera_controls()
                    print(f"[Camera] Re-applied control locks: {status}")
                elif key == ord("r"):
                    if recorder.is_recording:
                        saved_vid = recorder.stop()
                        print(f"[Recorder] Stopped recording -> {saved_vid}")
                    else:
                        rec_path = recorder.start(w, h)
                        print(f"[Recorder] Started recording -> {rec_path}")
    finally:
        if recorder.is_recording:
            saved_vid = recorder.stop()
            print(f"[Recorder] Finalized recording -> {saved_vid} ({recorder.frame_count} frames)")
        if not headless:
            cv2.destroyAllWindows()

    print(f"[Acquisition] Processed {processed} frames | Measured FPS: {last_fps:.1f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 1: Video Acquisition & Play-Area ROI Calibration")
    parser.add_argument("--source", type=str, default=None, help="Camera index (e.g. 0), video path, or 'synthetic'")
    parser.add_argument("--config", type=Path, default=DEFAULT_CALIBRATION_PATH, help="Calibration JSON path")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window (for testing/benchmarking)")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames to process in headless mode")
    parser.add_argument("--record", action="store_true", help="Automatically record raw frames during run")
    parser.add_argument("--output-dir", type=Path, default=Path("data/recordings"), help="Directory for recorded clips")
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
