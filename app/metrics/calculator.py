"""
MetricsCalculator — Stages 2–4 orchestrator.

Pipeline per frame (DetectionStatus.OK only):
  1. raw EAR from landmark points
  2. EarSmoother → EMA + rolling baseline
  3. BlinkDetector (smoothed EAR) → blink events + stats
  4. PerclosTracker (smoothed EAR) → PERCLOS + long closures
  5. FatigueScorer → rule-based fatigue score
  6. Assemble EyeMetrics
"""

import time
from typing import Optional

import cv2
import numpy as np

from app.landmarks.result import DetectionResult, DetectionStatus
from app.metrics.ear import compute_ear, EyeState
from app.metrics.blink import BlinkDetector
from app.metrics.perclos import PerclosTracker
from app.metrics.fatigue import FatigueScorer
from app.metrics.result import EyeMetrics, FatigueLevel
from app.temporal.smoother import EarSmoother

_STATE_LABEL = {
    EyeState.OPEN:             "open",
    EyeState.PARTIALLY_CLOSED: "partial",
    EyeState.CLOSED:           "closed",
}
_LEVEL_LABEL = {
    FatigueLevel.ALERT:    "ALERT",
    FatigueLevel.MILD:     "MILD",
    FatigueLevel.MODERATE: "MODERATE",
    FatigueLevel.SEVERE:   "SEVERE",
}


class MetricsCalculator:
    def __init__(
        self,
        ema_alpha: float               = 0.15,
        baseline_window_seconds: float = 4.0,
        long_blink_threshold_ms: float = 400.0,
    ):
        self._smoother = EarSmoother(alpha=ema_alpha,
                                     window_seconds=baseline_window_seconds)
        self._blink    = BlinkDetector(long_blink_threshold_ms=long_blink_threshold_ms)
        self._perclos  = PerclosTracker()
        self._fatigue  = FatigueScorer()

    # ─── public API ──────────────────────────────────────────────────────────

    def update(
        self,
        detection: DetectionResult,
        timestamp_ms: Optional[float] = None,
    ) -> Optional[EyeMetrics]:
        if detection.status != DetectionStatus.OK:
            return None
        if not (detection.right_eye and detection.left_eye):
            return None

        ts = timestamp_ms if timestamp_ms is not None else time.time() * 1_000.0

        # ── 1. raw EAR ────────────────────────────────────────────────────────
        ear_r_raw = compute_ear(detection.right_eye.ear_points)
        ear_l_raw = compute_ear(detection.left_eye.ear_points)

        # ── 2. smoothing ──────────────────────────────────────────────────────
        smooth = self._smoother.update(ear_l_raw, ear_r_raw, ts)

        # ── 3. blink detection ────────────────────────────────────────────────
        blink_event = self._blink.update(smooth.ema_avg, ts)
        blink_stats = self._blink.get_stats(ts)

        # ── 4. PERCLOS + long closures ────────────────────────────────────────
        self._perclos.update(smooth.ema_avg, ts)
        perclos_30 = self._perclos.get_perclos(ts, window_seconds=30.0)
        perclos_60 = self._perclos.get_perclos(ts, window_seconds=60.0)
        long_count = self._perclos.get_long_closure_count(ts, window_seconds=60.0)
        total_close_ms = self._perclos.get_total_closure_time_ms(ts, window_seconds=60.0)

        # ── 5. fatigue score ──────────────────────────────────────────────────
        fat = self._fatigue.score(
            perclos=perclos_60,
            long_closure_count=long_count,
            avg_blink_duration_ms=blink_stats.avg_blink_duration_ms,
        )

        # ── 6. assemble ───────────────────────────────────────────────────────
        return EyeMetrics(
            # raw
            ear_left=ear_l_raw,
            ear_right=ear_r_raw,
            ear_avg=smooth.raw_avg,
            # smoothed
            ear_left_smooth=smooth.ema_left,
            ear_right_smooth=smooth.ema_right,
            ear_avg_smooth=smooth.ema_avg,
            # baseline
            ear_baseline=smooth.baseline,
            ear_deviation=smooth.deviation,
            # state (from smoothed)
            state_left=EyeState.from_ear(smooth.ema_left),
            state_right=EyeState.from_ear(smooth.ema_right),
            # blink
            blink_detected=blink_event is not None,
            blink_duration_ms=blink_event.duration_ms if blink_event else 0.0,
            # blink stats
            blink_rate_30s=float(blink_stats.blink_count_30s),
            blink_rate_60s=blink_stats.blink_rate_60s,
            avg_blink_duration_ms=blink_stats.avg_blink_duration_ms,
            long_blink_count=blink_stats.long_blink_count,
            # PERCLOS
            perclos_30s=perclos_30,
            perclos_60s=perclos_60,
            # long closures
            long_closure_count_60s=long_count,
            total_closure_time_60s_ms=total_close_ms,
            # fatigue
            fatigue_score=fat.score,
            fatigue_level=fat.level,
        )

    def draw(self, frame: np.ndarray, metrics: Optional[EyeMetrics]) -> np.ndarray:
        if metrics is None:
            return frame

        h, w = frame.shape[:2]

        # ── EAR + state row ───────────────────────────────────────────────────
        ear_text = (
            f"EAR  L:{metrics.ear_left:.3f}→{metrics.ear_left_smooth:.3f}"
            f"[{_STATE_LABEL[metrics.state_left]}]  "
            f"R:{metrics.ear_right:.3f}→{metrics.ear_right_smooth:.3f}"
            f"[{_STATE_LABEL[metrics.state_right]}]"
        )
        cv2.putText(frame, ear_text,
                    (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (220, 220, 220), 1, cv2.LINE_AA)

        # ── Baseline + PERCLOS row ────────────────────────────────────────────
        cv2.putText(
            frame,
            f"base:{metrics.ear_baseline:.3f}  dev:{metrics.ear_deviation:+.3f}  "
            f"PERCLOS 30s:{metrics.perclos_30s:.2f}  60s:{metrics.perclos_60s:.2f}  "
            f"long:{metrics.long_closure_count_60s}",
            (10, h - 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (180, 180, 180), 1, cv2.LINE_AA)

        # ── Blink stats row ───────────────────────────────────────────────────
        cv2.putText(
            frame,
            f"Blinks  30s:{int(metrics.blink_rate_30s)}  "
            f"60s:{int(metrics.blink_rate_60s)}  "
            f"avgDur:{metrics.avg_blink_duration_ms:.0f}ms  "
            f"long:{metrics.long_blink_count}",
            (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (220, 220, 220), 1, cv2.LINE_AA)

        # ── Fatigue score bar (top-right) ──────────────────────────────────────
        self._draw_fatigue_bar(frame, metrics.fatigue_score, metrics.fatigue_level)

        # ── Blink flash ───────────────────────────────────────────────────────
        if metrics.blink_detected:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (255, 255, 0), 4)
            cv2.putText(frame, f"BLINK  {metrics.blink_duration_ms:.0f}ms",
                        (w // 2 - 70, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        return frame

    def reset(self):
        self._smoother.reset()
        self._blink.reset()
        self._perclos.reset()

    # ─── private ─────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_fatigue_bar(
        frame: np.ndarray,
        score: float,
        level: FatigueLevel,
    ):
        h, w = frame.shape[:2]
        bar_w, bar_h = 160, 18
        x0 = w - bar_w - 12
        y0 = 10

        # background
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (50, 50, 50), -1)
        # filled portion
        filled = int(bar_w * score)
        colour = level.colour_bgr
        if filled > 0:
            cv2.rectangle(frame, (x0, y0), (x0 + filled, y0 + bar_h), colour, -1)
        # border
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (160, 160, 160), 1)
        # label
        label = f"{_LEVEL_LABEL[level]}  {score:.2f}"
        cv2.putText(frame, label,
                    (x0, y0 + bar_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
