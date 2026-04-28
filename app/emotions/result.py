"""Result types for emotion detection."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class Emotion(Enum):
    NEUTRAL   = "neutral"
    HAPPY     = "happy"
    SAD       = "sad"
    SURPRISED = "surprised"
    ANGRY     = "angry"
    FEARFUL   = "fearful"


@dataclass
class EmotionResult:
    emotion:    Emotion
    confidence: float
    scores:     Dict[Emotion, float] = field(default_factory=dict)

    # Raw geometric features (useful for debugging / tuning thresholds)
    mar:        float = 0.0   # Mouth Aspect Ratio (openness)
    smile:      float = 0.0   # >0 = corners up (happy), <0 = corners down (sad)
    brow_raise: float = 0.0   # brow height above eye, normalised by face height
    brow_tilt:  float = 0.0   # (inner_brow_y − outer_brow_y) / face_h; >0 = angry slant
