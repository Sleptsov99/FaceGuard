"""Output types for eye and fatigue metrics (Stages 2–4)."""

from dataclasses import dataclass
from enum import Enum

from app.metrics.ear import EyeState


# ─── FatigueLevel lives here to avoid circular import ────────────────────────

class FatigueLevel(Enum):
    ALERT    = "alert"     # score < 0.25
    MILD     = "mild"      # 0.25 ≤ score < 0.50
    MODERATE = "moderate"  # 0.50 ≤ score < 0.75
    SEVERE   = "severe"    # score ≥ 0.75

    @classmethod
    def from_score(cls, score: float) -> "FatigueLevel":
        if score >= 0.75:
            return cls.SEVERE
        if score >= 0.50:
            return cls.MODERATE
        if score >= 0.25:
            return cls.MILD
        return cls.ALERT

    @property
    def colour_bgr(self) -> tuple:
        """OpenCV BGR colour for overlays."""
        return {
            FatigueLevel.ALERT:    (0, 220,   0),
            FatigueLevel.MODERATE: (0, 165, 255),
            FatigueLevel.MILD:     (0, 220, 220),
            FatigueLevel.SEVERE:   (0,   0, 220),
        }[self]


# ─── EyeMetrics ──────────────────────────────────────────────────────────────

@dataclass
class EyeMetrics:
    # ── Raw EAR (per-frame) ───────────────────────────────────────────────────
    ear_left:  float
    ear_right: float
    ear_avg:   float

    # ── EMA-smoothed EAR (Stage 3) ────────────────────────────────────────────
    ear_left_smooth:  float
    ear_right_smooth: float
    ear_avg_smooth:   float

    # ── Rolling baseline (Stage 3) ────────────────────────────────────────────
    ear_baseline:  float   # rolling mean of smoothed avg over last N seconds
    ear_deviation: float   # baseline − smooth_avg  (positive → eyes closing)

    # ── Eye state (from smoothed EAR) ────────────────────────────────────────
    state_left:  EyeState
    state_right: EyeState

    # ── Current blink ─────────────────────────────────────────────────────────
    blink_detected:    bool
    blink_duration_ms: float

    # ── Blink statistics (sliding window) ────────────────────────────────────
    blink_rate_30s:        float
    blink_rate_60s:        float
    avg_blink_duration_ms: float
    long_blink_count:      int     # blinks 400–800 ms in last 60 s

    # ── PERCLOS (Stage 4) ─────────────────────────────────────────────────────
    perclos_30s: float   # fraction of frames with EAR < closed_threshold in 30 s
    perclos_60s: float   # same for 60 s

    # ── Long closures (Stage 4) ───────────────────────────────────────────────
    long_closure_count_60s:    int     # closures > 500 ms in last 60 s
    total_closure_time_60s_ms: float  # cumulative closed time for those closures

    # ── Fatigue score (Stage 4) ───────────────────────────────────────────────
    fatigue_score: float        # [0, 1]
    fatigue_level: FatigueLevel
