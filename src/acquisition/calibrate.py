"""Play-area ROI calibration, camera lock verification, and raw clip recording CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.acquisition.camera import VideoStream
from src.acquisition.recorder import GroundTruthLogger, VideoRecorder
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
    clip_rolls: Optional[List[int]] = None,
    total_rolls: int = 0,
) -> np.ndarray:
    """Overlay ROI box, FPS, camera lock state, roll log, and key controls on a frame copy."""
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
    rolls_str = str(clip_rolls) if clip_rolls else "[]"
    cv2.putText(
        vis,
        f"Clip rolls: {rolls_str} | Total logged: {total_rolls}/100",
        (16, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        "Drag: ROI | [R] Rec | [1-6] Log Roll | [U] Undo | [L] Lock | [S] Save | [Q] Quit",
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
    gt_csv: Path = Path("data/ground_truth.csv"),
) -> int:
    cfg = load_config(config_path)
    if source is not None:
        cfg.source = int(source) if source.isdigit() else source

    selector = _ROISelector(cfg.roi)
    recorder = VideoRecorder(output_dir=output_dir, fps=cfg.target_fps)
    gt_logger = GroundTruthLogger(csv_path=gt_csv)
    clip_rolls: List[int] = []
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
            print(f"[GT] Ground-truth CSV: {gt_logger.csv_path} (existing rolls: {gt_logger.total_count})")

            if record:
                rec_path = recorder.start(stream.width, stream.height)
                clip_rolls.clear()
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

                hud = draw_hud(
                    frame,
                    selector.roi,
                    last_fps,
                    recorder.is_recording,
                    stream.lock_status,
                    selector,
                    clip_rolls=clip_rolls,
                    total_rolls=gt_logger.total_count,
                )
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
                        print(f"[Recorder] Stopped -> {recorder.stop()} (logged {len(clip_rolls)} rolls: {clip_rolls})")
                    else:
                        clip_rolls.clear()
                        print(f"[Recorder] Started -> {recorder.start(stream.width, stream.height)}")
                elif ord("1") <= key <= ord("6"):
                    val = key - ord("0")
                    if recorder.is_recording and recorder.video_path is not None:
                        roll_idx = len(clip_rolls) + 1
                        gt_logger.log_roll(recorder.video_path, val, recorder.frame_count, roll_idx)
                        clip_rolls.append(val)
                        print(
                            f"[GT] Logged {recorder.video_path.name} roll #{roll_idx} = {val} "
                            f"(frame {recorder.frame_count}) | Total: {gt_logger.total_count}"
                        )
                    else:
                        print("[GT] Start recording [R] before logging a roll.")
                elif key == ord("u"):
                    if recorder.is_recording and recorder.video_path is not None and clip_rolls:
                        undone = gt_logger.undo_last(recorder.video_path)
                        if undone is not None:
                            clip_rolls.pop()
                            print(f"[GT] Undid roll = {undone} | Clip rolls: {clip_rolls} | Total: {gt_logger.total_count}")
                    else:
                        print("[GT] Nothing to undo in current clip.")
    finally:
        if recorder.is_recording:
            print(f"[Recorder] Finalized -> {recorder.stop()} ({recorder.frame_count} frames, rolls: {clip_rolls})")
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
    parser.add_argument("--gt-csv", type=Path, default=Path("data/ground_truth.csv"), help="CSV path for ground-truth rolls")
    args = parser.parse_args()
    return run_calibration(
        source=args.source,
        config_path=args.config,
        headless=args.headless,
        max_frames=args.frames,
        record=args.record,
        output_dir=args.output_dir,
        gt_csv=args.gt_csv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
