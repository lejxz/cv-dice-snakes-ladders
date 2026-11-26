"""Step 6: Geometric 3x3 pip layout verification and multi-frame majority voting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from src.vision.pip_detector import PipDetection

# Canonical 3x3 occupancy patterns across all 90-degree planar rotations.
CANONICAL_PATTERNS: Dict[FrozenSet[Tuple[int, int]], int] = {
    frozenset({(1, 1)}): 1,
    frozenset({(0, 0), (2, 2)}): 2,
    frozenset({(0, 2), (2, 0)}): 2,
    frozenset({(0, 0), (1, 1), (2, 2)}): 3,
    frozenset({(0, 2), (1, 1), (2, 0)}): 3,
    frozenset({(0, 0), (0, 2), (2, 0), (2, 2)}): 4,
    frozenset({(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)}): 5,
    frozenset({(0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (2, 2)}): 6,
    frozenset({(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)}): 6,
}


@dataclass
class FrameVerdict:
    """Single-frame verification result combining dual-count and 3x3 layout checks."""

    value: Optional[int]
    count_ok: bool
    layout_ok: bool
    grid: np.ndarray
    reason: str


@dataclass
class RollResult:
    """Multi-frame majority voting result over K settled frames."""

    value: Optional[int]
    confidence: float
    accepted: bool
    total_frames: int
    support_frames: int


def verify_layout(
    centers: Sequence[Tuple[float, float]],
    face_shape: Tuple[int, int] = (120, 120),
    inner_margin: float = 0.0,
) -> Tuple[Optional[int], np.ndarray, str]:
    """Map pip centers onto a 3x3 grid and verify against canonical die patterns."""
    grid = np.zeros((3, 3), dtype=np.uint8)
    h, w = face_shape[:2]
    if h <= 0 or w <= 0 or not (0.0 <= inner_margin < 0.5):
        return None, grid, "INVALID_LAYOUT"

    span_x = w * (1.0 - 2.0 * inner_margin)
    span_y = h * (1.0 - 2.0 * inner_margin)
    offset_x = w * inner_margin
    offset_y = h * inner_margin

    occupied = set()
    for x, y in centers:
        nx = min(max((x - offset_x) / span_x, 0.0), 0.999999)
        ny = min(max((y - offset_y) / span_y, 0.0), 0.999999)
        col = int(3.0 * nx)
        row = int(3.0 * ny)
        cell = (row, col)
        if cell in occupied:
            return None, grid, "CELL_COLLISION"
        occupied.add(cell)
        grid[row, col] = 1

    value = CANONICAL_PATTERNS.get(frozenset(occupied))
    if value is None:
        return None, grid, "INVALID_LAYOUT"
    return value, grid, "VALID"


def verify_frame(
    pip_detection: PipDetection,
    use_blob: bool = True,
    use_cc: bool = True,
    use_grid_check: bool = True,
    inner_margin: float = 0.0,
) -> FrameVerdict:
    """Verify a single frame's PipDetection via count agreement and 3x3 layout."""
    empty_grid = np.zeros((3, 3), dtype=np.uint8)

    if use_blob and use_cc:
        if pip_detection.blob_count != pip_detection.cc_count:
            return FrameVerdict(None, False, False, empty_grid, "COUNT_MISMATCH")
        raw_count = pip_detection.blob_count
        centers = pip_detection.blob_centers
    elif use_blob:
        raw_count = pip_detection.blob_count
        centers = pip_detection.blob_centers
    elif use_cc:
        raw_count = pip_detection.cc_count
        centers = pip_detection.cc_centers
    else:
        return FrameVerdict(None, False, False, empty_grid, "COUNT_MISMATCH")

    if not (1 <= raw_count <= 6):
        return FrameVerdict(None, False, False, empty_grid, "OUT_OF_RANGE")

    if not use_grid_check:
        return FrameVerdict(raw_count, True, True, empty_grid, "VALID")

    face_shape = pip_detection.pip_mask.shape[:2]
    value, grid, reason = verify_layout(
        centers, face_shape=face_shape, inner_margin=inner_margin
    )
    if value is None or value != raw_count:
        return FrameVerdict(None, True, False, grid, reason)

    return FrameVerdict(value, True, True, grid, "VALID")


def vote_roll(
    verdicts: Sequence[FrameVerdict],
    min_confidence: float = 0.70,
) -> RollResult:
    """Aggregate K settled frame verdicts using majority voting and confidence gating."""
    total = len(verdicts)
    if total == 0:
        return RollResult(None, 0.0, False, 0, 0)

    valid_values = [v.value for v in verdicts if v.value is not None]
    if not valid_values:
        return RollResult(None, 0.0, False, total, 0)

    counts = Counter(valid_values).most_common()
    top_value, top_count = counts[0]
    if len(counts) > 1 and counts[1][1] == top_count:
        return RollResult(None, top_count / total, False, total, top_count)

    confidence = top_count / total
    if confidence < min_confidence:
        return RollResult(None, confidence, False, total, top_count)

    return RollResult(top_value, confidence, True, total, top_count)


class RollVoter:
    """Collects K settled FrameVerdicts and emits a voted RollResult."""

    def __init__(self, window_size: int = 12, min_confidence: float = 0.70) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.verdicts: List[FrameVerdict] = []

    @property
    def is_ready(self) -> bool:
        """True once window_size frames have been collected."""
        return len(self.verdicts) >= self.window_size

    def add(self, verdict: FrameVerdict) -> Optional[RollResult]:
        """Append a FrameVerdict and return RollResult once window_size is reached."""
        self.verdicts.append(verdict)
        if len(self.verdicts) < self.window_size:
            return None
        return vote_roll(self.verdicts, min_confidence=self.min_confidence)

    def reset(self) -> None:
        """Clear accumulated verdicts for the next roll event."""
        self.verdicts.clear()
