"""Classical die detection, contour filtering, and upright die-face warping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.vision.preprocessing import PreprocessedFrame


@dataclass
class DieDetection:
    """Result of detecting and cropping a single white die in a frame."""

    center_roi: Tuple[float, float]
    rect_roi: Tuple[Tuple[float, float], Tuple[float, float], float]
    box_points_roi: np.ndarray  # (4, 2) int32 corners in ROI coordinates
    face_gray: np.ndarray       # Upright grayscale crop from recognition_gray
    area_roi: float


def _order_box_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 box corners consistently as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]     # top-left
    ordered[2] = pts[np.argmax(s)]     # bottom-right
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    return ordered


def crop_upright_face(
    gray: np.ndarray,
    rect_roi: Tuple[Tuple[float, float], Tuple[float, float], float],
    face_size: int = 120,
    margin_ratio: float = 0.08,
) -> np.ndarray:
    """Warp the rotated die bounding box into an upright square grayscale patch."""
    box = _order_box_points(cv2.boxPoints(rect_roi))
    center = box.mean(axis=0)
    # Pull corners slightly inward to trim beveled die edges
    inset_box = center + (box - center) * (1.0 - 2.0 * margin_ratio)

    dst = np.array(
        [
            [0, 0],
            [face_size - 1, 0],
            [face_size - 1, face_size - 1],
            [0, face_size - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(inset_box.astype(np.float32), dst)
    return cv2.warpPerspective(gray, matrix, (face_size, face_size))


def detect_die(
    prep: PreprocessedFrame,
    min_area_roi: float = 900.0,
    max_area_roi: float = 40000.0,
    min_aspect_ratio: float = 0.75,
    min_rectangularity: float = 0.78,
    use_adaptive: bool = False,
    face_size: int = 120,
    margin_ratio: float = 0.08,
) -> Optional[DieDetection]:
    """Detect a single white die on a dark surface and return its upright face crop.

    Returns None if 0 or >1 valid die candidates are found (single-die ambiguity guard).
    """
    det = prep.detection_gray

    if use_adaptive:
        mask = cv2.adaptiveThreshold(
            det, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -15
        )
    else:
        _, mask = cv2.threshold(det, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Close dark pips near die edges, then open to remove small surface dust specks
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scale_sq = prep.scale * prep.scale
    min_area_det = min_area_roi * scale_sq
    max_area_det = max_area_roi * scale_sq

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area_det <= area <= max_area_det):
            continue

        rect = cv2.minAreaRect(cnt)
        (_, _), (w, h), _ = rect
        if w <= 0 or h <= 0:
            continue

        aspect = min(w, h) / max(w, h)
        rectangularity = area / (w * h)
        if aspect >= min_aspect_ratio and rectangularity >= min_rectangularity:
            candidates.append((rect, area))

    # Single-die guard: skip if missing or ambiguous (e.g., hand/multiple objects)
    if len(candidates) != 1:
        return None

    rect_det, area_det = candidates[0]
    rect_roi = prep.scale_rect_to_roi(rect_det)
    box_points_roi = np.round(cv2.boxPoints(rect_roi)).astype(np.int32)
    face_gray = crop_upright_face(
        prep.recognition_gray,
        rect_roi,
        face_size=face_size,
        margin_ratio=margin_ratio,
    )

    return DieDetection(
        center_roi=rect_roi[0],
        rect_roi=rect_roi,
        box_points_roi=box_points_roi,
        face_gray=face_gray,
        area_roi=area_det / scale_sq,
    )
