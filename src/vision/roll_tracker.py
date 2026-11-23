"""Temporal roll-event state machine to detect when a thrown die has stopped."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from src.vision.die_detector import DieDetection


class RollTracker:
    """Tracks die center movement to detect when a throw has settled."""

    def __init__(
        self,
        settle_frames: int = 10,
        settle_px: float = 2.0,
        move_px: float = 6.0,
        missing_reset_frames: int = 5,
    ) -> None:
        self.settle_frames = settle_frames
        self.settle_px = settle_px
        self.move_px = move_px
        self.missing_reset_frames = missing_reset_frames

        self.state: str = "IDLE"
        self.last_center: Optional[Tuple[float, float]] = None
        self.stable_count: int = 0
        self.missing_count: int = 0

    @property
    def is_settled(self) -> bool:
        """True when the die has stopped and is ready for pip reading."""
        return self.state == "SETTLED"

    def mark_done(self) -> None:
        """Latch current roll as completed until the die moves or is picked up."""
        self.state = "DONE"

    def reset(self) -> None:
        """Reset tracker to initial IDLE state."""
        self.state = "IDLE"
        self.last_center = None
        self.stable_count = 0
        self.missing_count = 0

    def update(self, detection: Optional[DieDetection]) -> str:
        """Update state with the current frame's DieDetection (or None) and return state."""
        if detection is None:
            self.missing_count += 1
            if self.missing_count >= self.missing_reset_frames:
                self.reset()
            return self.state

        self.missing_count = 0
        center = detection.center_roi

        if self.last_center is None:
            self.last_center = center
            self.stable_count = 1
            self.state = "SETTLED" if self.settle_frames <= 1 else "MOVING"
            return self.state

        dist = math.hypot(center[0] - self.last_center[0], center[1] - self.last_center[1])
        self.last_center = center

        if self.state == "DONE":
            if dist > self.move_px:
                self.state = "MOVING"
                self.stable_count = 0
            return self.state

        if dist <= self.settle_px:
            self.stable_count += 1
            if self.stable_count >= self.settle_frames:
                self.state = "SETTLED"
        else:
            self.stable_count = 0
            self.state = "MOVING"

        return self.state
