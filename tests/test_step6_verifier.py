"""Unit tests for Step 6: 3x3 Pip Layout Verification & Multi-Frame Majority Voting."""

from __future__ import annotations

import cv2
import numpy as np

from src.config import ROIConfig
from src.vision.die_detector import detect_die
from src.vision.pip_detector import PipDetection, detect_pips
from src.vision.preprocessing import preprocess_frame
from src.vision.roll_tracker import RollTracker
from src.vision.verifier import (
    CANONICAL_PATTERNS,
    FrameVerdict,
    RollVoter,
    verify_frame,
    verify_layout,
    vote_roll,
)

ALL_TEMPLATES = {
    "1": ([(60, 60)], 1),
    "2_main": ([(25, 25), (95, 95)], 2),
    "2_anti": ([(95, 25), (25, 95)], 2),
    "3_main": ([(25, 25), (60, 60), (95, 95)], 3),
    "3_anti": ([(95, 25), (60, 60), (25, 95)], 3),
    "4": ([(25, 25), (95, 25), (25, 95), (95, 95)], 4),
    "5": ([(25, 25), (95, 25), (60, 60), (25, 95), (95, 95)], 5),
    "6_cols": ([(25, 22), (25, 60), (25, 98), (95, 22), (95, 60), (95, 98)], 6),
    "6_rows": ([(22, 25), (60, 25), (98, 25), (22, 95), (60, 95), (98, 95)], 6),
}


def _make_face(pips: list[tuple[int, int]], radius: int = 8) -> np.ndarray:
    face = np.full((120, 120), 235, dtype=np.uint8)
    for cx, cy in pips:
        cv2.circle(face, (cx, cy), radius, 25, -1)
    return face


def test_all_nine_canonical_templates_verify() -> None:
    assert len(CANONICAL_PATTERNS) == 9
    for _, (pts, expected) in ALL_TEMPLATES.items():
        face = _make_face(pts)
        pip_det = detect_pips(face)
        verdict = verify_frame(pip_det)

        assert verdict.count_ok is True
        assert verdict.layout_ok is True
        assert verdict.value == expected
        assert verdict.reason == "VALID"


def test_glare_spot_on_four_rejected_not_misclassified_as_five() -> None:
    # 4 valid corners + 1 false glare pip in top-middle cell (0, 1)
    four_plus_glare = [(25, 25), (95, 25), (25, 95), (95, 95), (60, 22)]
    face = _make_face(four_plus_glare)
    pip_det = detect_pips(face)
    assert pip_det.count == 5  # Dual-count alone would think it's a 5

    verdict = verify_frame(pip_det)
    assert verdict.value is None
    assert verdict.count_ok is True
    assert verdict.layout_ok is False
    assert verdict.reason == "INVALID_LAYOUT"


def test_partial_occlusion_on_five_rejected() -> None:
    # 3 corners + center (missing bottom-right corner) -> 4 pips, invalid layout
    occluded_five = [(25, 25), (95, 25), (60, 60), (25, 95)]
    verdict = verify_frame(detect_pips(_make_face(occluded_five)))
    assert verdict.value is None
    assert verdict.layout_ok is False
    assert verdict.reason == "INVALID_LAYOUT"


def test_cell_collision_rejected() -> None:
    # Two pips in top-left cell (0, 0)
    value, _, reason = verify_layout([(20.0, 20.0), (30.0, 30.0)], face_shape=(120, 120))
    assert value is None
    assert reason == "CELL_COLLISION"


def test_count_mismatch_rejected() -> None:
    dummy_mask = np.zeros((120, 120), dtype=np.uint8)
    mismatch = PipDetection(
        blob_count=4,
        cc_count=5,
        blob_centers=[(25.0, 25.0), (95.0, 25.0), (25.0, 95.0), (95.0, 95.0)],
        cc_centers=[(25.0, 25.0), (95.0, 25.0), (25.0, 95.0), (95.0, 95.0), (60.0, 60.0)],
        count_agreed=False,
        pip_mask=dummy_mask,
    )
    verdict = verify_frame(mismatch)
    assert verdict.value is None
    assert verdict.count_ok is False
    assert verdict.reason == "COUNT_MISMATCH"


def test_ablation_flags() -> None:
    four_plus_glare = [(25, 25), (95, 25), (25, 95), (95, 95), (60, 22)]
    pip_det = detect_pips(_make_face(four_plus_glare))

    # Disabling grid check accepts raw count 5
    no_grid = verify_frame(pip_det, use_grid_check=False)
    assert no_grid.value == 5
    assert no_grid.reason == "VALID"

    # Single-detector mode works when one detector is disabled
    dummy_mask = np.zeros((120, 120), dtype=np.uint8)
    blob_only_det = PipDetection(
        blob_count=1,
        cc_count=2,
        blob_centers=[(60.0, 60.0)],
        cc_centers=[(60.0, 60.0), (25.0, 25.0)],
        count_agreed=False,
        pip_mask=dummy_mask,
    )
    assert verify_frame(blob_only_det, use_blob=True, use_cc=False).value == 1


def test_vote_roll_and_roll_voter() -> None:
    grid = np.zeros((3, 3), dtype=np.uint8)
    valid_5 = FrameVerdict(5, True, True, grid, "VALID")
    uncertain = FrameVerdict(None, True, False, grid, "INVALID_LAYOUT")
    valid_3 = FrameVerdict(3, True, True, grid, "VALID")
    valid_4 = FrameVerdict(4, True, True, grid, "VALID")

    # 10/12 support >= 0.70 -> accepted
    res_ok = vote_roll([valid_5] * 10 + [uncertain] * 2, min_confidence=0.70)
    assert res_ok.accepted is True
    assert res_ok.value == 5
    assert abs(res_ok.confidence - (10 / 12)) < 1e-6

    # 7/12 support < 0.70 -> rejected
    res_low = vote_roll([valid_5] * 7 + [uncertain] * 5, min_confidence=0.70)
    assert res_low.accepted is False
    assert res_low.value is None

    # Tie (6 vs 6) -> rejected
    res_tie = vote_roll([valid_3] * 6 + [valid_4] * 6, min_confidence=0.40)
    assert res_tie.accepted is False
    assert res_tie.value is None

    # RollVoter accumulates until window_size, then emits RollResult
    voter = RollVoter(window_size=5, min_confidence=0.70)
    for _ in range(4):
        assert voter.add(valid_5) is None
        assert not voter.is_ready
    result = voter.add(valid_5)
    assert voter.is_ready
    assert result is not None
    assert result.accepted is True
    assert result.value == 5

    voter.reset()
    assert not voter.is_ready
    assert len(voter.verdicts) == 0


def test_end_to_end_steps_2_to_6() -> None:
    frame = np.full((720, 1280, 3), 28, dtype=np.uint8)
    cx, cy, size = 600, 350, 90
    half = size / 2.0
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=np.float32)
    rad = np.deg2rad(28.0)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]], dtype=np.float32)
    pts = np.round(corners @ rot.T + np.array([cx, cy], dtype=np.float32)).astype(np.int32)
    cv2.fillPoly(frame, [pts], (238, 238, 238))

    pip_off = size * 0.24
    for px, py in [(-pip_off, -pip_off), (pip_off, -pip_off), (0.0, 0.0), (-pip_off, pip_off), (pip_off, pip_off)]:
        p = np.array([px, py], dtype=np.float32) @ rot.T + np.array([cx, cy], dtype=np.float32)
        cv2.circle(frame, (int(round(p[0])), int(round(p[1]))), 6, (25, 25, 25), -1)

    roi = ROIConfig(x=200, y=100, width=800, height=500)
    tracker = RollTracker(settle_frames=3)
    voter = RollVoter(window_size=4, min_confidence=0.75)

    final_result = None
    for _ in range(7):
        prep = preprocess_frame(frame, roi=roi, scale=0.5)
        det = detect_die(prep)
        assert det is not None
        tracker.update(det)
        if tracker.is_settled:
            verdict = verify_frame(detect_pips(det.face_gray))
            final_result = voter.add(verdict)
            if final_result is not None:
                tracker.mark_done()
                break

    assert final_result is not None
    assert final_result.accepted is True
    assert final_result.value == 5
    assert final_result.confidence == 1.0
    assert tracker.state == "DONE"
