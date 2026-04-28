"""
Emotion detector — rule-based geometric analysis of MediaPipe FaceMesh landmarks.

Features used (all in normalised landmark coordinates, 0–1):
  MAR        — Mouth Aspect Ratio: how open the mouth is
  smile      — corner raise relative to lip centre (>0 = smile, <0 = frown)
  brow_raise — average eyebrow height above upper eyelid, normalised by face height
  brow_tilt  — (inner_brow_y − outer_brow_y) / face_h
               positive  = inner corners LOWER than outer → angry/scowling slant
               negative  = inner corners HIGHER (fully raised brows)

Each feature is mapped to a soft score via sigmoid, then EMA-smoothed over time
to suppress per-frame jitter.
"""

from typing import Optional, Dict
import cv2
import numpy as np

from app.landmarks.result import DetectionResult
from app.emotions.result import Emotion, EmotionResult

# ─── MediaPipe landmark indices ───────────────────────────────────────────────

# Mouth
_MOUTH_R      = 61    # right corner
_MOUTH_L      = 291   # left corner
_LIP_IN_TOP   = 13    # inner upper lip centre
_LIP_IN_BOT   = 14    # inner lower lip centre
_LIP_OUT_TOP  = 0     # outer upper lip centre
_LIP_OUT_BOT  = 17    # outer lower lip centre

# Eyebrows
_BROW_R_MID   = 52    # right brow centre
_BROW_L_MID   = 282   # left brow centre
_BROW_R_IN    = 55    # right brow inner corner
_BROW_L_IN    = 285   # left brow inner corner
_BROW_R_OUT   = 46    # right brow outer corner
_BROW_L_OUT   = 276   # left brow outer corner

# Eyes
_EYE_R_TOP    = 159   # right upper eyelid (top)
_EYE_L_TOP    = 386   # left  upper eyelid (top)

# Face height references
_FOREHEAD     = 10
_CHIN         = 152

# ─── colours per emotion ──────────────────────────────────────────────────────

_EMOTION_COLOUR: Dict[Emotion, tuple] = {
    Emotion.NEUTRAL:   (180, 180, 180),
    Emotion.HAPPY:     (0,   220,   0),
    Emotion.SAD:       (200, 100,  50),
    Emotion.SURPRISED: (0,   200, 255),
    Emotion.ANGRY:     (0,    50, 220),
    Emotion.FEARFUL:   (180,   0, 200),
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(x)))


def _dist2d(a, b) -> float:
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


class EmotionDetector:
    """
    Detects facial emotion from an existing DetectionResult (uses raw_landmarks).

    Designed to be called once per frame immediately after LandmarkDetector.detect().
    """

    _EMA_ALPHA = 0.25   # 0 = no update, 1 = no history

    def __init__(self):
        self._smoothed: Dict[Emotion, float] = {e: 0.0 for e in Emotion}
        self._smoothed[Emotion.NEUTRAL] = 1.0

    # ─── public API ──────────────────────────────────────────────────────────

    def detect(self, detection: DetectionResult) -> Optional[EmotionResult]:
        """Return an EmotionResult for the current frame, or None if no face."""
        if detection.raw_landmarks is None:
            return None

        lm = detection.raw_landmarks.landmark

        face_h = abs(lm[_CHIN].y - lm[_FOREHEAD].y)
        if face_h < 0.01:
            return self._neutral()

        # ── MAR (Mouth Aspect Ratio) ──────────────────────────────────────
        mouth_w = _dist2d(lm[_MOUTH_R], lm[_MOUTH_L])
        lip_v   = abs(lm[_LIP_IN_TOP].y - lm[_LIP_IN_BOT].y)
        mar     = lip_v / max(mouth_w, 0.01)

        # ── smile score ───────────────────────────────────────────────────
        # y ↓ in image coords → corners above lip centre → smile (+), below → frown (-)
        lip_cy   = (lm[_LIP_OUT_TOP].y + lm[_LIP_OUT_BOT].y) * 0.5
        corner_y = (lm[_MOUTH_R].y     + lm[_MOUTH_L].y)      * 0.5
        smile    = (lip_cy - corner_y) / face_h

        # ── brow raise ────────────────────────────────────────────────────
        # How far above the upper eyelid each brow centre is, / face_h
        brow_raise = (
            (lm[_EYE_R_TOP].y - lm[_BROW_R_MID].y) +
            (lm[_EYE_L_TOP].y - lm[_BROW_L_MID].y)
        ) * 0.5 / face_h

        # ── brow tilt ─────────────────────────────────────────────────────
        # (inner corner y − outer corner y) / face_h
        # Neutral ≈ 0  |  Angry > 0 (inner lower)  |  Surprised < 0 (inner higher)
        brow_tilt = (
            (lm[_BROW_R_IN].y - lm[_BROW_R_OUT].y) +
            (lm[_BROW_L_IN].y - lm[_BROW_L_OUT].y)
        ) * 0.5 / face_h

        raw = self._raw_scores(mar, smile, brow_raise, brow_tilt)
        self._ema(raw)

        winner     = max(self._smoothed, key=self._smoothed.__getitem__)
        confidence = float(self._smoothed[winner])

        return EmotionResult(
            emotion=winner,
            confidence=confidence,
            scores=dict(self._smoothed),
            mar=mar,
            smile=smile,
            brow_raise=brow_raise,
            brow_tilt=brow_tilt,
        )

    def draw(self, frame: np.ndarray, result: Optional[EmotionResult]) -> np.ndarray:
        """Draw emotion label + raw features below the cv-state panel (y=190)."""
        if result is None:
            return frame
        colour = _EMOTION_COLOUR[result.emotion]
        cv2.putText(frame,
                    f"emotion: {result.emotion.value}  ({result.confidence:.2f})",
                    (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.60, colour, 1, cv2.LINE_AA)
        # Raw feature values — useful for threshold tuning
        cv2.putText(frame,
                    f"mar:{result.mar:.3f}  smile:{result.smile:+.3f}"
                    f"  brow_r:{result.brow_raise:.3f}  tilt:{result.brow_tilt:+.3f}",
                    (10, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)
        return frame

    # ─── private helpers ─────────────────────────────────────────────────────

    def _raw_scores(
        self,
        mar:        float,
        smile:      float,
        brow_raise: float,
        brow_tilt:  float,
    ) -> Dict[Emotion, float]:
        # ── intermediate activations ──────────────────────────────────────────
        # Target neutral-face values: mar≈0.03, smile≈0, brow_raise≈0.09, brow_tilt≈0
        mouth_open = _sigmoid((mar        - 0.12)  * 20)   # 0.14 closed → 0.93 wide open
        brow_up    = _sigmoid((brow_raise - 0.125) * 60)   # 0.11 normal → 0.71 raised
        angry_brow = _sigmoid((brow_tilt  - 0.02)  * 80)   # 0.17 neutral → 0.83 angry

        # ── emotion scores ────────────────────────────────────────────────────
        # Double-sigmoid for happy/sad: near zero → low score; above threshold → rises fast.
        # This lets the function stay low at neutral (smile≈0) and respond at ±0.015+.
        happy = _sigmoid((smile  - 0.012) * 70) * _sigmoid((smile  - 0.002) * 50)
        sad   = _sigmoid((-smile - 0.012) * 70) * _sigmoid((-smile - 0.002) * 50)

        surprised = mouth_open * brow_up

        # Angry: downward inner brow slant AND not smiling
        angry = angry_brow * (1.0 - _sigmoid((smile - 0.025) * 60))

        # Fearful: raised brows but mouth not wide open (that would be surprised)
        fearful = brow_up * (1.0 - 0.8 * mouth_open)

        # Neutral holds a hard floor of 0.35 so it always competes;
        # it wins only while no emotion exceeds ~0.35.
        combined = max(happy, sad, surprised, angry, fearful)
        neutral  = max(0.35, 1.0 - combined * 2.2)

        return {
            Emotion.NEUTRAL:   neutral,
            Emotion.HAPPY:     happy,
            Emotion.SAD:       sad,
            Emotion.SURPRISED: surprised,
            Emotion.ANGRY:     angry,
            Emotion.FEARFUL:   fearful,
        }

    def _ema(self, raw: Dict[Emotion, float]) -> None:
        a = self._EMA_ALPHA
        for e, v in raw.items():
            self._smoothed[e] = a * v + (1.0 - a) * self._smoothed[e]

    def _neutral(self) -> EmotionResult:
        return EmotionResult(
            emotion=Emotion.NEUTRAL,
            confidence=1.0,
            scores={e: (1.0 if e == Emotion.NEUTRAL else 0.0) for e in Emotion},
        )
