"""Tests for Stage 7: CV-state layer."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.landmarks.result import DetectionResult, DetectionStatus, HeadPose
from app.metrics.ear import EyeState
from app.metrics.result import EyeMetrics, FatigueLevel
from app.state.distraction import DistractionResult, DistractionReason
from app.state.cv_state import (
    CVStateEstimator, CVState,
    EyeStateCV, FatigueLevelCV, AttentionState,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _metrics(
    state_left=EyeState.OPEN, state_right=EyeState.OPEN,
    blink=False, blink_ms=0.0,
    fatigue_level=FatigueLevel.ALERT, fatigue_score=0.0,
) -> EyeMetrics:
    return EyeMetrics(
        ear_left=0.30, ear_right=0.30, ear_avg=0.30,
        ear_left_smooth=0.30, ear_right_smooth=0.30, ear_avg_smooth=0.30,
        ear_baseline=0.30, ear_deviation=0.0,
        state_left=state_left, state_right=state_right,
        blink_detected=blink, blink_duration_ms=blink_ms,
        blink_rate_30s=0.0, blink_rate_60s=0.0,
        avg_blink_duration_ms=0.0, long_blink_count=0,
        perclos_30s=0.0, perclos_60s=0.0,
        long_closure_count_60s=0, total_closure_time_60s_ms=0.0,
        fatigue_score=fatigue_score, fatigue_level=fatigue_level,
    )


def _distr(
    face_present=True,
    is_distracted=False,
    is_low_confidence=False,
    reason=DistractionReason.NONE,
    face_absent_ms=0.0,
) -> DistractionResult:
    return DistractionResult(
        face_present=face_present,
        face_absent_ms=face_absent_ms,
        face_absent_alert=False,
        yaw=0.0, pitch=0.0, roll=0.0,
        is_looking_away=is_distracted and reason == DistractionReason.LOOKING_AWAY,
        is_head_tilted=False,
        confidence=0.9 if face_present else 0.0,
        is_low_confidence=is_low_confidence,
        distraction_score=0.5 if is_distracted else 0.0,
        is_distracted=is_distracted,
        reason=reason,
    )


est = CVStateEstimator()


# ─── EyeStateCV ──────────────────────────────────────────────────────────────

class TestEyeStateCV(unittest.TestCase):
    def test_open_eyes(self):
        s = est.estimate(_metrics(), _distr())
        self.assertEqual(s.eye_state, EyeStateCV.OPEN)

    def test_blink_takes_priority_over_open(self):
        s = est.estimate(_metrics(blink=True, blink_ms=120.0), _distr())
        self.assertEqual(s.eye_state, EyeStateCV.BLINK)

    def test_closed_eye(self):
        s = est.estimate(_metrics(state_left=EyeState.CLOSED), _distr())
        self.assertEqual(s.eye_state, EyeStateCV.CLOSED)

    def test_partially_closed_is_closed(self):
        s = est.estimate(_metrics(state_right=EyeState.PARTIALLY_CLOSED), _distr())
        self.assertEqual(s.eye_state, EyeStateCV.CLOSED)

    def test_no_metrics_gives_unknown(self):
        s = est.estimate(None, _distr())
        self.assertEqual(s.eye_state, EyeStateCV.UNKNOWN)

    def test_blink_takes_priority_over_closed(self):
        s = est.estimate(
            _metrics(state_left=EyeState.CLOSED, blink=True, blink_ms=100.0),
            _distr(),
        )
        self.assertEqual(s.eye_state, EyeStateCV.BLINK)


# ─── FatigueLevelCV ──────────────────────────────────────────────────────────

class TestFatigueLevelCV(unittest.TestCase):
    def test_alert_maps_to_normal(self):
        s = est.estimate(_metrics(fatigue_level=FatigueLevel.ALERT), _distr())
        self.assertEqual(s.fatigue_level, FatigueLevelCV.NORMAL)

    def test_mild_maps_to_warning(self):
        s = est.estimate(_metrics(fatigue_level=FatigueLevel.MILD), _distr())
        self.assertEqual(s.fatigue_level, FatigueLevelCV.WARNING)

    def test_moderate_maps_to_warning(self):
        s = est.estimate(_metrics(fatigue_level=FatigueLevel.MODERATE), _distr())
        self.assertEqual(s.fatigue_level, FatigueLevelCV.WARNING)

    def test_severe_maps_to_high(self):
        s = est.estimate(_metrics(fatigue_level=FatigueLevel.SEVERE), _distr())
        self.assertEqual(s.fatigue_level, FatigueLevelCV.HIGH)

    def test_no_metrics_gives_normal(self):
        s = est.estimate(None, _distr())
        self.assertEqual(s.fatigue_level, FatigueLevelCV.NORMAL)


# ─── AttentionState ──────────────────────────────────────────────────────────

class TestAttentionState(unittest.TestCase):
    def test_face_present_not_distracted(self):
        s = est.estimate(_metrics(), _distr())
        self.assertEqual(s.attention, AttentionState.ATTENTIVE)

    def test_face_absent(self):
        s = est.estimate(None, _distr(face_present=False, face_absent_ms=500.0))
        self.assertEqual(s.attention, AttentionState.ABSENT)

    def test_low_confidence(self):
        s = est.estimate(_metrics(), _distr(is_low_confidence=True))
        self.assertEqual(s.attention, AttentionState.LOW_CONFIDENCE)

    def test_distracted(self):
        s = est.estimate(
            _metrics(),
            _distr(is_distracted=True, reason=DistractionReason.LOOKING_AWAY),
        )
        self.assertEqual(s.attention, AttentionState.POSSIBLY_DISTRACTED)

    def test_absent_takes_priority_over_low_confidence(self):
        s = est.estimate(
            None,
            _distr(face_present=False, is_low_confidence=True),
        )
        self.assertEqual(s.attention, AttentionState.ABSENT)


# ─── CVState is frozen dataclass ─────────────────────────────────────────────

class TestCVStateImmutable(unittest.TestCase):
    def test_is_frozen(self):
        s = est.estimate(_metrics(), _distr())
        with self.assertRaises((AttributeError, TypeError)):
            s.eye_state = EyeStateCV.UNKNOWN  # type: ignore

    def test_all_fields_present(self):
        s = est.estimate(_metrics(), _distr())
        self.assertIsInstance(s.eye_state,     EyeStateCV)
        self.assertIsInstance(s.fatigue_level, FatigueLevelCV)
        self.assertIsInstance(s.attention,     AttentionState)


# ─── draw smoke test ─────────────────────────────────────────────────────────

class TestCVStateDraw(unittest.TestCase):
    def test_draw_does_not_crash(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for eye in EyeStateCV:
            for fat in FatigueLevelCV:
                for att in AttentionState:
                    s = CVState(eye_state=eye, fatigue_level=fat, attention=att)
                    est.draw(frame.copy(), s)   # must not raise


if __name__ == "__main__":
    unittest.main()
