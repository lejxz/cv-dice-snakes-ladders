"""Unit tests for Step 4: Roll Event Detection (RollTracker)."""

from __future__ import annotations

import numpy as np

from src.vision.die_detector import DieDetection
from src.vision.roll_tracker import RollTracker


def _det(cx: float, cy: float) -> DieDetection:
    return DieDetection(
        center_roi=(cx, cy),
        rect_roi=((cx, cy), (80.0, 80.0), 0.0),
        box_points_roi=np.zeros((4, 2), dtype=np.int32),
        face_gray=np.zeros((120, 120), dtype=np.uint8),
        area_roi=6400.0,
    )


def test_moving_die_never_settled() -> None:
    tracker = RollTracker(settle_frames=10, settle_px=2.0)
    for i in range(15):
        state = tracker.update(_det(100.0 + i * 10.0, 200.0))
        assert state == "MOVING"
        assert not tracker.is_settled


def test_settles_after_n_stable_frames() -> None:
    tracker = RollTracker(settle_frames=10, settle_px=2.0)
    # Moving phase (ends at x=148.0)
    for i in range(5):
        assert tracker.update(_det(100.0 + i * 12.0, 200.0)) == "MOVING"

    # Landing frame at x=200.0 (jumped from 148.0 -> still MOVING)
    assert tracker.update(_det(200.0, 200.0)) == "MOVING"

    # 9 consecutive stationary frames (with tiny 0.5 px jitter < 2.0 px)
    for i in range(9):
        state = tracker.update(_det(200.0 + (i % 2) * 0.5, 200.0))
        assert state == "MOVING"
        assert not tracker.is_settled

    # 10th consecutive stable frame -> SETTLED
    assert tracker.update(_det(200.0, 200.0)) == "SETTLED"
    assert tracker.is_settled


def test_brief_pause_during_bounce_does_not_settle() -> None:
    tracker = RollTracker(settle_frames=10, settle_px=2.0)
    tracker.update(_det(100.0, 200.0))

    # Pause for 8 frames (< 10), then bounce
    for _ in range(8):
        assert tracker.update(_det(100.0, 200.0)) == "MOVING"

    assert tracker.update(_det(115.0, 205.0)) == "MOVING"
    assert tracker.stable_count == 0
    assert not tracker.is_settled


def test_mark_done_and_re_roll_unlock() -> None:
    tracker = RollTracker(settle_frames=5, settle_px=2.0, move_px=6.0, missing_reset_frames=3)

    for _ in range(5):
        tracker.update(_det(300.0, 300.0))
    assert tracker.is_settled

    # Latch as DONE; staying still should keep DONE
    tracker.mark_done()
    for _ in range(5):
        assert tracker.update(_det(300.5, 300.0)) == "DONE"

    # Moving > move_px unlocks to MOVING for the next roll
    assert tracker.update(_det(320.0, 300.0)) == "MOVING"

    # Picking up die (None for 3 frames) resets to IDLE
    for _ in range(3):
        tracker.update(None)
    assert tracker.state == "IDLE"
