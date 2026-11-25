"""Unit tests for Step 5: Pip Segmentation & Dual-Method Counting."""

from __future__ import annotations

import cv2
import numpy as np

from src.config import ROIConfig
from src.vision.die_detector import detect_die
from src.vision.pip_detector import detect_pips
from src.vision.preprocessing import preprocess_frame

CANONICAL_PIPS = {
    1: [(60, 60)],
    2: [(30, 30), (90, 90)],
    3: [(30, 30), (60, 60), (90, 90)],
    4: [(30, 30), (90, 30), (30, 90), (90, 90)],
    5: [(30, 30), (90, 30), (60, 60), (30, 90), (90, 90)],
    6: [(30, 26), (30, 60), (30, 94), (90, 26), (90, 60), (90, 94)],
}


def _make_face(pips: list[tuple[int, int]], radius: int = 8) -> np.ndarray:
    """Create a synthetic 120x120 upright white die face with dark pips."""
    face = np.full((120, 120), 235, dtype=np.uint8)
    for cx, cy in pips:
        cv2.circle(face, (cx, cy), radius, 25, -1)
    return face


def test_detect_pips_all_six_faces() -> None:
    for value, expected_pts in CANONICAL_PIPS.items():
        face = _make_face(expected_pts)
        res = detect_pips(face)

        assert res.blob_count == value
        assert res.cc_count == value
        assert res.count_agreed is True
        assert res.count == value

        for ex, ey in expected_pts:
            assert any(abs(bx - ex) <= 1.5 and abs(by - ey) <= 1.5 for bx, by in res.blob_centers)
            assert any(abs(cx - ex) <= 1.5 and abs(cy - ey) <= 1.5 for cx, cy in res.cc_centers)


def test_border_artifacts_cleared() -> None:
    face = _make_face(CANONICAL_PIPS[4])
    # Dark shadow strips along borders should be flood-filled away
    face[:5, :] = 20
    face[:, -4:] = 20

    res = detect_pips(face)
    assert res.blob_count == 4
    assert res.cc_count == 4
    assert res.count_agreed is True


def test_noise_and_specular_highlight_handled() -> None:
    face = _make_face(CANONICAL_PIPS[3])
    # 1x1 / 2x2 dark dust specks (removed by opening)
    face[15:17, 70:72] = 20
    face[100:101, 20:21] = 20
    # Tiny bright specular pinhole inside center pip (filled by closing)
    face[60, 60] = 240

    res = detect_pips(face)
    assert res.blob_count == 3
    assert res.cc_count == 3
    assert res.count_agreed is True


def test_elongated_scratch_rejected() -> None:
    face = _make_face(CANONICAL_PIPS[2])
    # Elongated dark scratch (fails circularity & aspect ratio)
    cv2.line(face, (25, 95), (75, 95), 25, thickness=3)

    res = detect_pips(face)
    assert res.blob_count == 2
    assert res.cc_count == 2
    assert res.count_agreed is True


def test_blank_and_out_of_range_faces_not_agreed() -> None:
    blank = np.full((120, 120), 235, dtype=np.uint8)
    res_blank = detect_pips(blank)
    assert res_blank.blob_count == 0
    assert res_blank.cc_count == 0
    assert res_blank.count_agreed is False
    assert res_blank.count is None

    seven_pips = _make_face(CANONICAL_PIPS[6] + [(60, 60)])
    res_seven = detect_pips(seven_pips)
    assert res_seven.blob_count == 7
    assert res_seven.cc_count == 7
    assert res_seven.count_agreed is False
    assert res_seven.count is None


def test_end_to_end_with_rotated_die() -> None:
    frame = np.full((720, 1280, 3), 28, dtype=np.uint8)
    cx, cy, size = 600, 350, 90
    half = size / 2.0
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=np.float32)
    rad = np.deg2rad(35.0)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]], dtype=np.float32)
    pts = np.round(corners @ rot.T + np.array([cx, cy], dtype=np.float32)).astype(np.int32)
    cv2.fillPoly(frame, [pts], (238, 238, 238))

    pip_off = size * 0.24
    for px, py in [(-pip_off, -pip_off), (pip_off, -pip_off), (0.0, 0.0), (-pip_off, pip_off), (pip_off, pip_off)]:
        p = np.array([px, py], dtype=np.float32) @ rot.T + np.array([cx, cy], dtype=np.float32)
        cv2.circle(frame, (int(round(p[0])), int(round(p[1]))), 6, (25, 25, 25), -1)

    prep = preprocess_frame(frame, roi=ROIConfig(x=200, y=100, width=800, height=500), scale=0.5)
    det = detect_die(prep)
    assert det is not None

    pip_res = detect_pips(det.face_gray)
    assert pip_res.blob_count == 5
    assert pip_res.cc_count == 5
    assert pip_res.count_agreed is True
    assert pip_res.count == 5
