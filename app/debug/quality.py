"""
Quality flags — Stage 9.

Detect per-frame conditions that may reduce metric reliability:
  NO_FACE        — face not detected
  LOW_CONFIDENCE — detection confidence below gate
  BAD_ANGLE      — extreme head pose, EAR unreliable
  FACE_TOO_SMALL — face too far from camera
  LOW_EAR        — smoothed EAR suspiciously low (glasses / poor light)
  PARTIAL_OCCL   — only one eye reliably detected (asymmetric EAR)
"""

from enum import Enum
from typing import FrozenSet, Optional

from app.landmarks.result import DetectionResult, DetectionStatus
from app.metrics.result import EyeMetrics


class QualityFlag(Enum):
    NO_FACE        = "no_face"
    LOW_CONFIDENCE = "low_confidence"
    BAD_ANGLE      = "bad_angle"
    FACE_TOO_SMALL = "face_too_small"
    LOW_EAR        = "low_ear"       # may indicate glasses / occlusion / dark room
    PARTIAL_OCCL   = "partial_occl"  # asymmetric EAR > threshold


_CONFIDENCE_GATE = 0.35
_LOW_EAR_GATE    = 0.15   # EAR below this even when "open" → suspect
_ASYM_GATE       = 0.08   # |ear_left - ear_right| above this → partial occlusion


def compute_quality_flags(
    detection: DetectionResult,
    metrics:   Optional[EyeMetrics],
) -> FrozenSet[QualityFlag]:
    flags: set[QualityFlag] = set()

    status = detection.status

    if status == DetectionStatus.NO_FACE:
        flags.add(QualityFlag.NO_FACE)
        return frozenset(flags)

    if status == DetectionStatus.FACE_TOO_SMALL:
        flags.add(QualityFlag.FACE_TOO_SMALL)

    if status == DetectionStatus.BAD_ANGLE:
        flags.add(QualityFlag.BAD_ANGLE)

    if detection.confidence < _CONFIDENCE_GATE:
        flags.add(QualityFlag.LOW_CONFIDENCE)

    if metrics is not None:
        if metrics.ear_avg_smooth < _LOW_EAR_GATE:
            flags.add(QualityFlag.LOW_EAR)

        asym = abs(metrics.ear_left_smooth - metrics.ear_right_smooth)
        if asym > _ASYM_GATE:
            flags.add(QualityFlag.PARTIAL_OCCL)

    return frozenset(flags)


def flags_to_str(flags: FrozenSet[QualityFlag]) -> str:
    """Compact string for CSV, e.g. 'low_confidence|bad_angle'."""
    if not flags:
        return "ok"
    return "|".join(sorted(f.value for f in flags))
