"""Computer vision pipeline modules for die detection, tracking, and pip verification."""

from src.vision.die_detector import DieDetection, crop_upright_face, detect_die
from src.vision.pip_detector import (
    PipDetection,
    detect_pips,
    detect_pips_blob,
    detect_pips_cc,
    segment_pips,
)
from src.vision.preprocessing import PreprocessedFrame, preprocess_frame
from src.vision.roll_tracker import RollTracker
from src.vision.verifier import (
    CANONICAL_PATTERNS,
    FrameVerdict,
    RollResult,
    RollVoter,
    verify_frame,
    verify_layout,
    vote_roll,
)

__all__ = [
    "CANONICAL_PATTERNS",
    "DieDetection",
    "FrameVerdict",
    "PipDetection",
    "PreprocessedFrame",
    "RollResult",
    "RollTracker",
    "RollVoter",
    "crop_upright_face",
    "detect_die",
    "detect_pips",
    "detect_pips_blob",
    "detect_pips_cc",
    "preprocess_frame",
    "segment_pips",
    "verify_frame",
    "verify_layout",
    "vote_roll",
]
