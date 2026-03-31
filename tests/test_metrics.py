"""Tests for Stage 2 eye metrics."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.metrics.ear import compute_ear, EyeState, EAR_OPEN_THRESHOLD, EAR_CLOSED_THRESHOLD
from app.metrics.blink import BlinkDetector
from app.metrics.calculator import MetricsCalculator
from app.landmarks.result import DetectionResult, DetectionStatus, EyeLandmarks, HeadPose


# ─── helpers ─────────────────────────────────────────────────────────────────

def make_ear_points(ear: float) -> np.ndarray:
    """
    Build a (6, 2) array that yields exactly the requested EAR.

    Layout (symmetric eye, width=100):
      p1=(0,0)   outer corner
      p2=(25,+v) upper outer
      p3=(75,+v) upper inner
      p4=(100,0) inner corner
      p5=(75,-v) lower inner
      p6=(25,-v) lower outer

    EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
         = (2v + 2v) / (2*100) = v/50
    → v = ear * 50
    """
    v = float(ear) * 50.0
    return np.array([
        [  0.0,  0.0],
        [ 25.0,    v],
        [ 75.0,    v],
        [100.0,  0.0],
        [ 75.0,   -v],
        [ 25.0,   -v],
    ], dtype=np.float32)


def make_detection(ear_l: float, ear_r: float,
                   status: DetectionStatus = DetectionStatus.OK) -> DetectionResult:
    return DetectionResult(
        status=status,
        confidence=0.9,
        right_eye=EyeLandmarks(contour=np.zeros((16, 2)), ear_points=make_ear_points(ear_r)),
        left_eye =EyeLandmarks(contour=np.zeros((16, 2)), ear_points=make_ear_points(ear_l)),
        head_pose=HeadPose(yaw=0.0, pitch=0.0, roll=0.0),
    )


# ─── EAR tests ───────────────────────────────────────────────────────────────

class TestComputeEAR(unittest.TestCase):
    def test_open_eye(self):
        ear = compute_ear(make_ear_points(0.30))
        self.assertAlmostEqual(ear, 0.30, places=5)

    def test_closed_eye(self):
        ear = compute_ear(make_ear_points(0.0))
        self.assertAlmostEqual(ear, 0.0, places=5)

    def test_arbitrary_value(self):
        for target in (0.10, 0.20, 0.35):
            with self.subTest(target=target):
                self.assertAlmostEqual(compute_ear(make_ear_points(target)), target, places=5)

    def test_degenerate_zero_width(self):
        # p1 == p4 → denominator is 0 → should return 0 without error
        pts = np.zeros((6, 2), dtype=np.float32)
        self.assertEqual(compute_ear(pts), 0.0)


class TestEyeState(unittest.TestCase):
    def test_open(self):
        self.assertEqual(EyeState.from_ear(EAR_OPEN_THRESHOLD), EyeState.OPEN)
        self.assertEqual(EyeState.from_ear(0.40), EyeState.OPEN)

    def test_partially_closed(self):
        mid = (EAR_OPEN_THRESHOLD + EAR_CLOSED_THRESHOLD) / 2
        self.assertEqual(EyeState.from_ear(mid), EyeState.PARTIALLY_CLOSED)

    def test_closed(self):
        self.assertEqual(EyeState.from_ear(EAR_CLOSED_THRESHOLD - 0.01), EyeState.CLOSED)
        self.assertEqual(EyeState.from_ear(0.0), EyeState.CLOSED)

    def test_custom_thresholds(self):
        self.assertEqual(EyeState.from_ear(0.30, open_thresh=0.35), EyeState.PARTIALLY_CLOSED)


# ─── BlinkDetector tests ─────────────────────────────────────────────────────

class TestBlinkDetector(unittest.TestCase):
    def setUp(self):
        self.det = BlinkDetector(
            ear_threshold=0.20,
            min_blink_duration_ms=50,
            max_blink_duration_ms=800,
            long_blink_threshold_ms=400,
        )

    def _ts(self, offset_ms: float = 0.0) -> float:
        return 1_000_000.0 + offset_ms   # fixed base timestamp

    def test_no_blink_above_threshold(self):
        for i in range(10):
            event = self.det.update(0.30, self._ts(i * 33))
            self.assertIsNone(event)
        stats = self.det.get_stats(self._ts(330))
        self.assertEqual(stats.blink_count_60s, 0)

    def test_single_blink_detected(self):
        # EAR drops below threshold for 100 ms, then recovers
        self.assertIsNone(self.det.update(0.30, self._ts(0)))
        self.assertIsNone(self.det.update(0.10, self._ts(100)))   # eye closes
        event = self.det.update(0.30, self._ts(200))              # eye opens
        self.assertIsNotNone(event)
        self.assertAlmostEqual(event.duration_ms, 100.0, delta=1.0)

    def test_blink_too_short_ignored(self):
        self.det.update(0.10, self._ts(0))
        event = self.det.update(0.30, self._ts(10))  # only 10 ms < min=50 ms
        self.assertIsNone(event)
        self.assertEqual(self.det.get_stats(self._ts(10)).blink_count_60s, 0)

    def test_blink_timeout_cancelled(self):
        # Eye "closes" but never recovers within max_blink_duration
        self.det.update(0.10, self._ts(0))
        # 900 ms later (> max=800), still closed — next update should cancel
        self.det.update(0.10, self._ts(900))
        # Eye finally opens — should NOT record a blink
        event = self.det.update(0.30, self._ts(950))
        self.assertIsNone(event)

    def test_multiple_blinks_counted(self):
        ts = 0.0
        for _ in range(5):
            self.det.update(0.10, ts);       ts += 150
            self.det.update(0.30, ts);       ts += 500   # 650 ms per cycle
        stats = self.det.get_stats(ts)
        self.assertEqual(stats.blink_count_60s, 5)

    def test_blink_exits_30s_window(self):
        # Record a blink, then advance time by 31 s
        self.det.update(0.10, self._ts(0))
        self.det.update(0.30, self._ts(100))
        stats_now = self.det.get_stats(self._ts(100))
        self.assertEqual(stats_now.blink_count_30s, 1)

        stats_later = self.det.get_stats(self._ts(31_000))
        self.assertEqual(stats_later.blink_count_30s, 0)
        self.assertEqual(stats_later.blink_count_60s, 1)  # still in 60s window

    def test_long_blink_counted(self):
        self.det.update(0.10, self._ts(0))
        self.det.update(0.30, self._ts(500))   # 500 ms > long_threshold=400
        stats = self.det.get_stats(self._ts(500))
        self.assertEqual(stats.long_blink_count, 1)

    def test_short_blink_not_long(self):
        self.det.update(0.10, self._ts(0))
        self.det.update(0.30, self._ts(100))
        stats = self.det.get_stats(self._ts(100))
        self.assertEqual(stats.long_blink_count, 0)

    def test_avg_duration(self):
        # Two blinks: 100 ms and 300 ms → avg = 200 ms
        self.det.update(0.10, self._ts(0))
        self.det.update(0.30, self._ts(100))
        self.det.update(0.10, self._ts(600))
        self.det.update(0.30, self._ts(900))
        stats = self.det.get_stats(self._ts(900))
        self.assertAlmostEqual(stats.avg_blink_duration_ms, 200.0, delta=1.0)

    def test_reset_clears_state(self):
        self.det.update(0.10, self._ts(0))
        self.det.update(0.30, self._ts(100))
        self.det.reset()
        stats = self.det.get_stats(self._ts(200))
        self.assertEqual(stats.blink_count_60s, 0)
        self.assertFalse(self.det._in_blink)


# ─── MetricsCalculator tests ─────────────────────────────────────────────────

class TestMetricsCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = MetricsCalculator()

    def test_returns_none_when_no_face(self):
        result = DetectionResult(status=DetectionStatus.NO_FACE)
        self.assertIsNone(self.calc.update(result))

    def test_returns_none_when_bad_angle(self):
        det = make_detection(0.30, 0.30, status=DetectionStatus.BAD_ANGLE)
        self.assertIsNone(self.calc.update(det))

    def test_returns_none_when_multiple_faces(self):
        det = make_detection(0.30, 0.30, status=DetectionStatus.MULTIPLE_FACES)
        self.assertIsNone(self.calc.update(det))

    def test_ear_values_correct(self):
        metrics = self.calc.update(make_detection(0.30, 0.28))
        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(metrics.ear_left,  0.30, places=4)
        self.assertAlmostEqual(metrics.ear_right, 0.28, places=4)
        self.assertAlmostEqual(metrics.ear_avg,   0.29, places=4)

    def test_eye_states_correct(self):
        metrics = self.calc.update(make_detection(0.30, 0.10))
        self.assertEqual(metrics.state_left,  EyeState.OPEN)
        self.assertEqual(metrics.state_right, EyeState.CLOSED)

    def test_blink_detected_on_recovery(self):
        # Use alpha=0.5 so EMA reacts fast enough for a 3-frame test sequence.
        # Default alpha=0.15 is intentionally slow (noise rejection) and would
        # require many closed frames to cross the threshold — tested in test_temporal.py.
        calc = MetricsCalculator(ema_alpha=0.5)
        base = 1_000_000.0

        # Frame 1: bootstrap — EMA = 0.30
        m1 = calc.update(make_detection(0.30, 0.30), timestamp_ms=base)
        self.assertFalse(m1.blink_detected)

        # Frame 2: eyes close — EMA = 0.5*0.0 + 0.5*0.30 = 0.15 < 0.23 → blink starts
        m2 = calc.update(make_detection(0.0, 0.0), timestamp_ms=base + 100)
        self.assertFalse(m2.blink_detected)   # blink not complete yet

        # Frame 3: eyes reopen — EMA = 0.5*0.30 + 0.5*0.15 = 0.225 still < 0.23
        m3 = calc.update(make_detection(0.30, 0.30), timestamp_ms=base + 250)
        self.assertFalse(m3.blink_detected)

        # Frame 4: EMA = 0.5*0.30 + 0.5*0.225 = 0.263 > 0.23 → blink completes
        m4 = calc.update(make_detection(0.30, 0.30), timestamp_ms=base + 280)
        self.assertTrue(m4.blink_detected)
        self.assertGreater(m4.blink_duration_ms, 0)

    def test_no_blink_when_always_open(self):
        for _ in range(10):
            m = self.calc.update(make_detection(0.35, 0.35))
        self.assertFalse(m.blink_detected)
        self.assertEqual(m.blink_rate_60s, 0.0)

    def test_draw_does_not_crash(self):
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        metrics = self.calc.update(make_detection(0.30, 0.30))
        out = self.calc.draw(frame.copy(), metrics)
        self.assertEqual(out.shape, frame.shape)

    def test_draw_none_metrics_no_crash(self):
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = self.calc.draw(frame.copy(), None)
        self.assertEqual(out.shape, frame.shape)

    def test_reset_clears_blink_history(self):
        self.calc.update(make_detection(0.10, 0.10))
        self.calc.update(make_detection(0.30, 0.30))
        self.calc.reset()
        m = self.calc.update(make_detection(0.30, 0.30))
        self.assertEqual(m.blink_rate_60s, 0.0)


if __name__ == "__main__":
    unittest.main()
