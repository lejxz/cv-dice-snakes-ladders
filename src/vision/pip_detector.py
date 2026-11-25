"""Step 5: Pip segmentation and dual-method counting (SimpleBlobDetector + CCA)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class PipDetection:
    """Result of dual-method pip detection on an upright die face."""

    blob_count: int
    cc_count: int
    blob_centers: List[Tuple[float, float]]
    cc_centers: List[Tuple[float, float]]
    count_agreed: bool
    pip_mask: np.ndarray

    @property
    def count(self) -> Optional[int]:
        """Agreed pip count (1-6), or None if methods disagree or out of range."""
        return self.blob_count if self.count_agreed else None


def _clear_border(mask: np.ndarray) -> np.ndarray:
    """Remove foreground components touching any border of a binary mask."""
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return mask

    cleaned = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    for x in range(w):
        if cleaned[0, x] != 0:
            cv2.floodFill(cleaned, ff_mask, (x, 0), 0)
        if cleaned[h - 1, x] != 0:
            cv2.floodFill(cleaned, ff_mask, (x, h - 1), 0)

    for y in range(h):
        if cleaned[y, 0] != 0:
            cv2.floodFill(cleaned, ff_mask, (0, y), 0)
        if cleaned[y, w - 1] != 0:
            cv2.floodFill(cleaned, ff_mask, (w - 1, y), 0)

    return cleaned


def segment_pips(face_gray: np.ndarray, min_contrast: int = 30) -> np.ndarray:
    """Segment dark pips as white foreground on a black binary mask."""
    if face_gray.size == 0:
        return np.zeros_like(face_gray, dtype=np.uint8)

    if int(face_gray.max()) - int(face_gray.min()) < min_contrast:
        return np.zeros(face_gray.shape[:2], dtype=np.uint8)

    _, mask = cv2.threshold(face_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return _clear_border(mask)


def detect_pips_blob(
    pip_mask: np.ndarray,
    min_area: float = 40.0,
    max_area: float = 900.0,
    min_circularity: float = 0.65,
    min_convexity: float = 0.80,
    min_inertia: float = 0.50,
) -> List[Tuple[float, float]]:
    """Detect pip centers using cv2.SimpleBlobDetector on a binary pip mask."""
    if pip_mask.size == 0 or not np.any(pip_mask):
        return []

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 127
    params.maxThreshold = 255
    params.thresholdStep = 128
    params.minRepeatability = 1
    params.minDistBetweenBlobs = 4.0

    params.filterByColor = True
    params.blobColor = 255

    params.filterByArea = True
    params.minArea = float(min_area)
    params.maxArea = float(max_area)

    params.filterByCircularity = True
    params.minCircularity = float(min_circularity)

    params.filterByConvexity = True
    params.minConvexity = float(min_convexity)

    params.filterByInertia = True
    params.minInertiaRatio = float(min_inertia)

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(pip_mask)
    return [(float(kp.pt[0]), float(kp.pt[1])) for kp in keypoints]


def detect_pips_cc(
    pip_mask: np.ndarray,
    min_area: float = 40.0,
    max_area: float = 900.0,
    min_aspect: float = 0.65,
    max_aspect: float = 1.45,
    min_circularity: float = 0.65,
) -> List[Tuple[float, float]]:
    """Detect pip centers using 8-connected components and shape filtering."""
    if pip_mask.size == 0 or not np.any(pip_mask):
        return []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        pip_mask, connectivity=8
    )

    centers: List[Tuple[float, float]] = []
    for label_id in range(1, num_labels):
        area = float(stats[label_id, cv2.CC_STAT_AREA])
        if not (min_area <= area <= max_area):
            continue

        w = float(stats[label_id, cv2.CC_STAT_WIDTH])
        h = float(stats[label_id, cv2.CC_STAT_HEIGHT])
        if h <= 0 or not (min_aspect <= (w / h) <= max_aspect):
            continue

        comp_mask = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        perimeter = cv2.arcLength(contours[0], closed=True)
        if perimeter <= 0:
            continue

        circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        cx, cy = centroids[label_id]
        centers.append((float(cx), float(cy)))

    return centers


def detect_pips(
    face_gray: np.ndarray,
    min_contrast: int = 30,
    min_area: float = 40.0,
    max_area: float = 900.0,
    min_circularity: float = 0.65,
) -> PipDetection:
    """Segment pips and count them independently via SimpleBlobDetector and CCA."""
    pip_mask = segment_pips(face_gray, min_contrast=min_contrast)
    blob_centers = detect_pips_blob(
        pip_mask,
        min_area=min_area,
        max_area=max_area,
        min_circularity=min_circularity,
    )
    cc_centers = detect_pips_cc(
        pip_mask,
        min_area=min_area,
        max_area=max_area,
        min_circularity=min_circularity,
    )

    blob_count = len(blob_centers)
    cc_count = len(cc_centers)
    count_agreed = (blob_count == cc_count) and (1 <= blob_count <= 6)

    return PipDetection(
        blob_count=blob_count,
        cc_count=cc_count,
        blob_centers=blob_centers,
        cc_centers=cc_centers,
        count_agreed=count_agreed,
        pip_mask=pip_mask,
    )
