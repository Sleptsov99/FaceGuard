"""
Distraction tracker — Stage 5.

Tracks two practical distraction signals:
  1. Face absence      — user left the camera field of view
  2. Head pose deviation — user is looking away or head is excessively tilted

Confidence gating:
  When a reliable assessment is not possible (face too small, very low
  detection confidence) the result carries is_low_confidence=True so
  downstream consumers can choose to discard or hold the last known value
  rather than act on unreliable data.

  Note: BAD_ANGLE status means the eyes are unreliable but the head pose
  IS reliable (that's how we detected the bad angle).  BAD_ANGLE is treated
  as "looking away" here, not as low-confidence.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from app.landmarks.result import DetectionResult, DetectionStatus


# ─── result types ────────────────────────────────────────────────────────────

class DistractionReason(Enum):
    NONE           = "none"            # frontal face, high confidence
    FACE_ABSENT    = "face_absent"     # no face detected
    LOOKING_AWAY   = "looking_away"    # |yaw| or |pitch| above threshold
    HEAD_TILTED    = "head_tilted"     # |roll| above threshold
    LOW_CONFIDENCE = "low_confidence"  # face present but data unreliable


@dataclass
class DistractionResult:
    # ── face presence ─────────────────────────────────────────────────────────
    face_present:     bool
    face_absent_ms:   float   # ms since face disappeared; 0 when present
    face_absent_alert: bool   # absent_ms ≥ absence_alert_threshold

    # ── head pose (degrees; 0.0 when face absent) ─────────────────────────────
    yaw:            float
    pitch:          float
    roll:           float
    is_looking_away: bool    # |yaw| or |pitch| > threshold
    is_head_tilted:  bool    # |roll| > roll_threshold

    # ── confidence gate ───────────────────────────────────────────────────────
    confidence:       float
    is_low_confidence: bool   # True → eye/fatigue data unreliable

    # ── overall distraction ───────────────────────────────────────────────────
    distraction_score: float          # [0, 1]
    is_distracted:     bool
    reason:            DistractionReason


# ─── tracker ─────────────────────────────────────────────────────────────────

class DistractionTracker:
    """
    Accepts every DetectionResult (not just OK) to track presence and pose.

    Parameters
    ----------
    absence_alert_seconds : float
        Alert if face has been absent for this many seconds.
    yaw_threshold, pitch_threshold : float
        Degrees beyond which head is considered "looking away".
    roll_threshold : float
        Degrees of tilt beyond which head is "tilted".
    confidence_gate : float
        Detection confidence below this → is_low_confidence=True.
    """

    def __init__(
        self,
        absence_alert_seconds: float = 3.0,
        yaw_threshold:   float = 30.0,
        pitch_threshold: float = 30.0,
        roll_threshold:  float = 20.0,
        confidence_gate: float = 0.30,
    ):
        self.absence_alert_ms = absence_alert_seconds * 1_000.0
        self.yaw_threshold    = yaw_threshold
        self.pitch_threshold  = pitch_threshold
        self.roll_threshold   = roll_threshold
        self.confidence_gate  = confidence_gate

        # Personal baseline (updated by apply_calibration)
        self._yaw_mean:   float = 0.0
        self._pitch_mean: float = 0.0
        self._roll_mean:  float = 0.0

        self._face_absent_since_ms: Optional[float] = None

    # ─── public API ──────────────────────────────────────────────────────────

    def update(
        self,
        detection: DetectionResult,
        timestamp_ms: Optional[float] = None,
    ) -> DistractionResult:
        """Process every frame regardless of DetectionStatus."""
        if timestamp_ms is None:
            timestamp_ms = time.time() * 1_000.0

        # ── face presence ──────────────────────────────────────────────────────
        face_present = detection.face_found   # True for OK, BAD_ANGLE, MULTIPLE_FACES

        if face_present:
            self._face_absent_since_ms = None
        else:
            if self._face_absent_since_ms is None:
                self._face_absent_since_ms = timestamp_ms

        absent_ms = (
            timestamp_ms - self._face_absent_since_ms
            if self._face_absent_since_ms is not None else 0.0
        )
        absent_alert = absent_ms >= self.absence_alert_ms

        # ── confidence gate ───────────────────────────────────────────────────
        # Face too small → landmarks unreliable for everything
        # Very low confidence → unreliable
        # BAD_ANGLE is NOT low-confidence (head pose is still valid)
        confidence = detection.confidence
        is_low_confidence = (
            not face_present
            or detection.status == DetectionStatus.FACE_TOO_SMALL
            or confidence < self.confidence_gate
        )

        # ── head pose ─────────────────────────────────────────────────────────
        pose  = detection.head_pose
        yaw   = pose.yaw   if pose else 0.0
        pitch = pose.pitch if pose else 0.0
        roll  = pose.roll  if pose else 0.0

        # Deviation from personal baseline (0.0 before calibration → same as raw angle)
        yaw_dev   = abs(yaw   - self._yaw_mean)
        pitch_dev = abs(pitch - self._pitch_mean)
        roll_dev  = abs(roll  - self._roll_mean)

        is_looking_away = (
            face_present and pose is not None
            and (yaw_dev > self.yaw_threshold or pitch_dev > self.pitch_threshold)
        )
        is_head_tilted = (
            face_present and pose is not None
            and roll_dev > self.roll_threshold
        )

        # ── distraction score + reason ────────────────────────────────────────
        if not face_present:
            score  = min(absent_ms / self.absence_alert_ms, 1.0)
            reason = DistractionReason.FACE_ABSENT
        elif is_low_confidence:
            score  = 0.0
            reason = DistractionReason.LOW_CONFIDENCE
        elif is_looking_away:
            score  = self._pose_score(yaw_dev, pitch_dev, roll_dev)
            reason = DistractionReason.LOOKING_AWAY
        elif is_head_tilted:
            score  = self._pose_score(yaw_dev, pitch_dev, roll_dev)
            reason = DistractionReason.HEAD_TILTED
        else:
            score  = 0.0
            reason = DistractionReason.NONE

        is_distracted = reason in (
            DistractionReason.FACE_ABSENT,
            DistractionReason.LOOKING_AWAY,
            DistractionReason.HEAD_TILTED,
        )

        return DistractionResult(
            face_present=face_present,
            face_absent_ms=absent_ms,
            face_absent_alert=absent_alert,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            is_looking_away=is_looking_away,
            is_head_tilted=is_head_tilted,
            confidence=confidence,
            is_low_confidence=is_low_confidence,
            distraction_score=score,
            is_distracted=is_distracted,
            reason=reason,
        )

    def draw(self, frame: np.ndarray, result: DistractionResult) -> np.ndarray:
        """Draw distraction overlay (presence dot + score bar + reason text)."""
        h, w = frame.shape[:2]

        # ── presence dot + absence timer (top-left, below detector overlay) ───
        dot_colour = (0, 200, 0) if result.face_present else (0, 0, 200)
        cv2.circle(frame, (14, 82), 6, dot_colour, -1)

        if not result.face_present:
            absent_s = result.face_absent_ms / 1_000.0
            alert_colour = (0, 0, 220) if result.face_absent_alert else (0, 165, 255)
            cv2.putText(frame, f"ABSENT  {absent_s:.1f}s",
                        (26, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                        alert_colour, 2, cv2.LINE_AA)
        else:
            reason_text = result.reason.value if result.is_distracted else "on-task"
            colour = (0, 200, 0) if not result.is_distracted else (0, 165, 255)
            if result.reason == DistractionReason.FACE_ABSENT:
                colour = (0, 0, 220)
            cv2.putText(frame, f"{reason_text}  conf:{result.confidence:.2f}",
                        (26, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        colour, 1, cv2.LINE_AA)

        # ── distraction score bar (below fatigue bar, top-right) ──────────────
        bar_w, bar_h = 160, 14
        x0 = w - bar_w - 12
        y0 = 46
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (50, 50, 50), -1)
        filled = int(bar_w * result.distraction_score)
        bar_colour = (0, 200, 0) if result.distraction_score < 0.3 else (
            (0, 165, 255) if result.distraction_score < 0.7 else (0, 0, 220)
        )
        if filled > 0:
            cv2.rectangle(frame, (x0, y0), (x0 + filled, y0 + bar_h), bar_colour, -1)
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (160, 160, 160), 1)
        cv2.putText(frame,
                    f"distract {result.distraction_score:.2f}",
                    (x0, y0 + bar_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, bar_colour, 1, cv2.LINE_AA)

        # ── DISTRACTED banner (centre) ────────────────────────────────────────
        if result.is_distracted and not result.face_present:
            cv2.putText(frame, "! LOOK AT SCREEN",
                        (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 220), 2, cv2.LINE_AA)

        return frame

    def apply_calibration(self, profile: "CalibrationProfile"):  # type: ignore[name-defined]
        """Apply personal head-pose thresholds from a calibration session.

        DistractionTracker compares |yaw - yaw_mean| against yaw_threshold,
        so we store the personal mean and use it in _pose_score().
        """
        self._yaw_mean      = profile.yaw_mean
        self._pitch_mean    = profile.pitch_mean
        self._roll_mean     = profile.roll_mean
        self.yaw_threshold  = profile.yaw_threshold
        self.pitch_threshold = profile.pitch_threshold
        self.roll_threshold  = profile.roll_threshold

    def reset(self):
        self._face_absent_since_ms = None

    # ─── private ─────────────────────────────────────────────────────────────

    def _pose_score(self, yaw_dev: float, pitch_dev: float, roll_dev: float) -> float:
        """
        Normalized [0, 1] deviation beyond the threshold.
        Inputs are already |angle - personal_mean|.
        Score=0 at threshold, score=1 at 2× threshold.
        """
        yaw_excess   = max(0.0, yaw_dev   - self.yaw_threshold)
        pitch_excess = max(0.0, pitch_dev - self.pitch_threshold)
        roll_excess  = max(0.0, roll_dev  - self.roll_threshold)

        yaw_s   = min(yaw_excess   / self.yaw_threshold,   1.0)
        pitch_s = min(pitch_excess / self.pitch_threshold, 1.0)
        roll_s  = min(roll_excess  / self.roll_threshold,  1.0)

        return max(yaw_s, pitch_s, roll_s)
