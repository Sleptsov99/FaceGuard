"""Tests for Stage 5: distraction tracking."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.landmarks.result import DetectionResult, DetectionStatus, HeadPose
from app.state.distraction import DistractionTracker, DistractionReason


# ─── helpers ─────────────────────────────────────────────────────────────────

def _ts(offset_ms: float = 0.0) -> float:
    return 1_000_000.0 + offset_ms


def _detection(
    status: DetectionStatus = DetectionStatus.OK,
    confidence: float = 0.9,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
) -> DetectionResult:
    pose = HeadPose(yaw=yaw, pitch=pitch, roll=roll) if status != DetectionStatus.NO_FACE else None
    return DetectionResult(
        status=status,
        confidence=confidence,
        head_pose=pose,
    )


def _no_face() -> DetectionResult:
    return DetectionResult(status=DetectionStatus.NO_FACE, confidence=0.0)


# ─── face presence tests ──────────────────────────────────────────────────────

class TestFacePresence(unittest.TestCase):
    def setUp(self):
        self.tracker = DistractionTracker(absence_alert_seconds=3.0)

    def test_face_present_ok(self):
        r = self.tracker.update(_detection(), _ts(0))
        self.assertTrue(r.face_present)
        self.assertEqual(r.face_absent_ms, 0.0)
        self.assertFalse(r.face_absent_alert)

    def test_no_face_starts_timer(self):
        r = self.tracker.update(_no_face(), _ts(0))
        self.assertFalse(r.face_present)
        self.assertAlmostEqual(r.face_absent_ms, 0.0, delta=1.0)

    def test_absence_duration_accumulates(self):
        self.tracker.update(_no_face(), _ts(0))
        r = self.tracker.update(_no_face(), _ts(1_000))
        self.assertAlmostEqual(r.face_absent_ms, 1_000.0, delta=1.0)

    def test_absence_alert_fires_at_threshold(self):
        self.tracker.update(_no_face(), _ts(0))
        r_before = self.tracker.update(_no_face(), _ts(2_999))
        r_at     = self.tracker.update(_no_face(), _ts(3_000))
        self.assertFalse(r_before.face_absent_alert)
        self.assertTrue(r_at.face_absent_alert)

    def test_timer_resets_on_face_return(self):
        self.tracker.update(_no_face(), _ts(0))
        self.tracker.update(_no_face(), _ts(1_000))
        r = self.tracker.update(_detection(), _ts(2_000))
        self.assertTrue(r.face_present)
        self.assertEqual(r.face_absent_ms, 0.0)
        self.assertFalse(r.face_absent_alert)

    def test_face_found_for_bad_angle(self):
        """BAD_ANGLE still has a face in frame — should be present."""
        r = self.tracker.update(_detection(status=DetectionStatus.BAD_ANGLE), _ts(0))
        self.assertTrue(r.face_present)

    def test_reason_is_face_absent_when_missing(self):
        r = self.tracker.update(_no_face(), _ts(0))
        self.assertEqual(r.reason, DistractionReason.FACE_ABSENT)
        self.assertTrue(r.is_distracted)


# ─── head pose tests ──────────────────────────────────────────────────────────

class TestHeadPose(unittest.TestCase):
    def setUp(self):
        self.tracker = DistractionTracker(
            yaw_threshold=30.0, pitch_threshold=30.0, roll_threshold=20.0
        )

    def test_frontal_not_distracted(self):
        r = self.tracker.update(_detection(yaw=0, pitch=0, roll=0), _ts(0))
        self.assertFalse(r.is_distracted)
        self.assertEqual(r.reason, DistractionReason.NONE)
        self.assertAlmostEqual(r.distraction_score, 0.0, places=5)

    def test_yaw_beyond_threshold_is_distracted(self):
        r = self.tracker.update(_detection(yaw=35.0), _ts(0))
        self.assertTrue(r.is_looking_away)
        self.assertTrue(r.is_distracted)
        self.assertEqual(r.reason, DistractionReason.LOOKING_AWAY)

    def test_pitch_beyond_threshold_is_distracted(self):
        r = self.tracker.update(_detection(pitch=-35.0), _ts(0))
        self.assertTrue(r.is_looking_away)
        self.assertEqual(r.reason, DistractionReason.LOOKING_AWAY)

    def test_roll_beyond_threshold_is_tilted(self):
        r = self.tracker.update(_detection(roll=25.0), _ts(0))
        self.assertTrue(r.is_head_tilted)
        self.assertEqual(r.reason, DistractionReason.HEAD_TILTED)

    def test_within_threshold_not_distracted(self):
        r = self.tracker.update(_detection(yaw=29.9, pitch=0, roll=19.9), _ts(0))
        self.assertFalse(r.is_distracted)

    def test_bad_angle_status_is_looking_away(self):
        """BAD_ANGLE is set when yaw/pitch exceed LandmarkDetector threshold."""
        r = self.tracker.update(
            _detection(status=DetectionStatus.BAD_ANGLE, yaw=35.0),
            _ts(0)
        )
        self.assertTrue(r.is_looking_away)
        self.assertTrue(r.is_distracted)


# ─── distraction score tests ──────────────────────────────────────────────────

class TestDistractionScore(unittest.TestCase):
    def setUp(self):
        self.tracker = DistractionTracker(
            absence_alert_seconds=2.0,
            yaw_threshold=30.0,
        )

    def test_score_zero_when_frontal(self):
        r = self.tracker.update(_detection(yaw=0), _ts(0))
        self.assertAlmostEqual(r.distraction_score, 0.0, places=5)

    def test_score_zero_at_exactly_threshold(self):
        r = self.tracker.update(_detection(yaw=30.0), _ts(0))
        # Just at threshold → excess = 0 → score = 0 (not looking away yet)
        self.assertAlmostEqual(r.distraction_score, 0.0, places=5)

    def test_score_half_at_1_5x_threshold(self):
        r = self.tracker.update(_detection(yaw=45.0), _ts(0))
        # excess = 15, threshold = 30 → score = 0.5
        self.assertAlmostEqual(r.distraction_score, 0.5, places=5)

    def test_score_saturates_at_2x_threshold(self):
        r = self.tracker.update(_detection(yaw=60.0), _ts(0))
        self.assertAlmostEqual(r.distraction_score, 1.0, places=5)

    def test_score_capped_beyond_2x(self):
        r = self.tracker.update(_detection(yaw=90.0), _ts(0))
        self.assertAlmostEqual(r.distraction_score, 1.0, places=5)

    def test_absent_score_ramps_to_one(self):
        self.tracker.update(_no_face(), _ts(0))
        # halfway to alert threshold (1 s of 2 s)
        r = self.tracker.update(_no_face(), _ts(1_000))
        self.assertAlmostEqual(r.distraction_score, 0.5, delta=0.02)

    def test_absent_score_saturates_at_alert_threshold(self):
        self.tracker.update(_no_face(), _ts(0))
        r = self.tracker.update(_no_face(), _ts(2_000))
        self.assertAlmostEqual(r.distraction_score, 1.0, delta=0.02)

    def test_score_in_range(self):
        for yaw in [0, 15, 30, 45, 60, 90]:
            r = self.tracker.update(_detection(yaw=float(yaw)), _ts(0))
            self.assertGreaterEqual(r.distraction_score, 0.0)
            self.assertLessEqual(r.distraction_score, 1.0)


# ─── confidence gate tests ────────────────────────────────────────────────────

class TestConfidenceGate(unittest.TestCase):
    def setUp(self):
        self.tracker = DistractionTracker(confidence_gate=0.30)

    def test_high_confidence_not_gated(self):
        r = self.tracker.update(_detection(confidence=0.9), _ts(0))
        self.assertFalse(r.is_low_confidence)

    def test_low_confidence_gated(self):
        r = self.tracker.update(_detection(confidence=0.20), _ts(0))
        self.assertTrue(r.is_low_confidence)
        self.assertEqual(r.reason, DistractionReason.LOW_CONFIDENCE)
        self.assertFalse(r.is_distracted)

    def test_face_too_small_is_low_confidence(self):
        det = DetectionResult(
            status=DetectionStatus.FACE_TOO_SMALL,
            confidence=0.8,   # high confidence, but face too small
            head_pose=HeadPose(yaw=0, pitch=0, roll=0),
        )
        r = self.tracker.update(det, _ts(0))
        self.assertTrue(r.is_low_confidence)

    def test_bad_angle_not_low_confidence(self):
        """BAD_ANGLE has reliable head pose → confidence gate doesn't fire.
        The distraction itself IS reported (looking away) — that's expected."""
        r = self.tracker.update(
            _detection(status=DetectionStatus.BAD_ANGLE, confidence=0.8, yaw=35.0),
            _ts(0),
        )
        self.assertFalse(r.is_low_confidence)
        self.assertNotEqual(r.reason, DistractionReason.LOW_CONFIDENCE)

    def test_no_face_always_low_confidence(self):
        r = self.tracker.update(_no_face(), _ts(0))
        self.assertTrue(r.is_low_confidence)

    def test_low_confidence_score_is_zero(self):
        r = self.tracker.update(_detection(confidence=0.1), _ts(0))
        self.assertAlmostEqual(r.distraction_score, 0.0, places=5)


# ─── draw smoke test ──────────────────────────────────────────────────────────

class TestDistractionDraw(unittest.TestCase):
    def setUp(self):
        self.tracker = DistractionTracker()

    def _frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_draw_frontal_no_crash(self):
        r = self.tracker.update(_detection(), _ts(0))
        out = self.tracker.draw(self._frame(), r)
        self.assertEqual(out.shape, (480, 640, 3))

    def test_draw_absent_no_crash(self):
        r = self.tracker.update(_no_face(), _ts(0))
        out = self.tracker.draw(self._frame(), r)
        self.assertEqual(out.shape, (480, 640, 3))

    def test_draw_looking_away_no_crash(self):
        r = self.tracker.update(_detection(yaw=45.0), _ts(0))
        out = self.tracker.draw(self._frame(), r)
        self.assertEqual(out.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
