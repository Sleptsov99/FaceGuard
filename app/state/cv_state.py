"""
CV-state layer — Stage 7.

Translates raw metrics into three high-level CV states:
  - EyeStateCV    — what are the eyes doing right now?
  - FatigueLevelCV — how fatigued is the user?
  - AttentionState — is the user paying attention?

These are CV-interpretations, not final alerts.
Downstream consumers decide what to do with them.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from app.metrics.ear import EyeState
from app.metrics.result import EyeMetrics, FatigueLevel
from app.state.distraction import DistractionResult


# ─── state enums ─────────────────────────────────────────────────────────────

class EyeStateCV(Enum):
    OPEN    = "open"
    CLOSED  = "closed"
    BLINK   = "blink"
    UNKNOWN = "unknown"


class FatigueLevelCV(Enum):
    NORMAL  = "normal"
    WARNING = "warning"
    HIGH    = "high"


class AttentionState(Enum):
    ATTENTIVE           = "attentive"
    POSSIBLY_DISTRACTED = "possibly_distracted"
    ABSENT              = "absent"
    LOW_CONFIDENCE      = "low_confidence"


# ─── result ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CVState:
    eye_state:     EyeStateCV
    fatigue_level: FatigueLevelCV
    attention:     AttentionState


# ─── estimator ───────────────────────────────────────────────────────────────

class CVStateEstimator:
    """
    Stateless CV-state estimator.

    Call estimate() every frame with the latest EyeMetrics and
    DistractionResult to get a CVState snapshot.
    """

    def estimate(
        self,
        metrics:     Optional[EyeMetrics],
        distraction: DistractionResult,
    ) -> CVState:
        return CVState(
            eye_state=self._eye_state(metrics),
            fatigue_level=self._fatigue_level(metrics),
            attention=self._attention(distraction),
        )

    def draw(self, frame: np.ndarray, state: CVState) -> np.ndarray:
        """Draw a compact CV-state panel in the top-left corner."""
        _draw_cv_state(frame, state)
        return frame

    # ─── private ─────────────────────────────────────────────────────────────

    @staticmethod
    def _eye_state(metrics: Optional[EyeMetrics]) -> EyeStateCV:
        if metrics is None:
            return EyeStateCV.UNKNOWN
        if metrics.blink_detected:
            return EyeStateCV.BLINK
        if (metrics.state_left  in (EyeState.CLOSED, EyeState.PARTIALLY_CLOSED)
                or metrics.state_right in (EyeState.CLOSED, EyeState.PARTIALLY_CLOSED)):
            return EyeStateCV.CLOSED
        return EyeStateCV.OPEN

    @staticmethod
    def _fatigue_level(metrics: Optional[EyeMetrics]) -> FatigueLevelCV:
        if metrics is None:
            return FatigueLevelCV.NORMAL
        level = metrics.fatigue_level
        if level == FatigueLevel.ALERT:
            return FatigueLevelCV.NORMAL
        if level in (FatigueLevel.MILD, FatigueLevel.MODERATE):
            return FatigueLevelCV.WARNING
        return FatigueLevelCV.HIGH  # SEVERE

    @staticmethod
    def _attention(distraction: DistractionResult) -> AttentionState:
        if not distraction.face_present:
            return AttentionState.ABSENT
        if distraction.is_low_confidence:
            return AttentionState.LOW_CONFIDENCE
        if distraction.is_distracted:
            return AttentionState.POSSIBLY_DISTRACTED
        return AttentionState.ATTENTIVE


# ─── drawing helper ──────────────────────────────────────────────────────────

_EYE_COLOUR = {
    EyeStateCV.OPEN:    (0, 220, 0),
    EyeStateCV.CLOSED:  (0, 165, 255),
    EyeStateCV.BLINK:   (255, 220, 0),
    EyeStateCV.UNKNOWN: (120, 120, 120),
}
_FATIGUE_COLOUR = {
    FatigueLevelCV.NORMAL:  (0, 220, 0),
    FatigueLevelCV.WARNING: (0, 165, 255),
    FatigueLevelCV.HIGH:    (0, 0, 220),
}
_ATTENTION_COLOUR = {
    AttentionState.ATTENTIVE:           (0, 220, 0),
    AttentionState.POSSIBLY_DISTRACTED: (0, 165, 255),
    AttentionState.ABSENT:              (0, 0, 220),
    AttentionState.LOW_CONFIDENCE:      (120, 120, 120),
}


def _draw_cv_state(frame: np.ndarray, state: CVState):
    """Three labelled rows: eye / fatigue / attention."""
    rows = [
        ("eye",      state.eye_state.value,     _EYE_COLOUR[state.eye_state]),
        ("fatigue",  state.fatigue_level.value,  _FATIGUE_COLOUR[state.fatigue_level]),
        ("attention", state.attention.value,     _ATTENTION_COLOUR[state.attention]),
    ]

    x, y0, dy = 10, 120, 20
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1

    for i, (label, value, colour) in enumerate(rows):
        y = y0 + i * dy
        cv2.putText(frame, f"{label}: {value}",
                    (x, y), font, scale, colour, thick, cv2.LINE_AA)
