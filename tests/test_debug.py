"""Tests for Stage 9: debug / validation layer."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.landmarks.result import DetectionResult, DetectionStatus, HeadPose
from app.metrics.ear import EyeState
from app.metrics.result import EyeMetrics, FatigueLevel
from app.state.distraction import DistractionResult, DistractionReason
from app.state.cv_state import CVState, EyeStateCV, FatigueLevelCV, AttentionState
from app.debug.quality import QualityFlag, compute_quality_flags, flags_to_str
from app.debug.logger import MetricsLogger
from app.debug.plotter import MetricsPlotter


# ─── helpers ─────────────────────────────────────────────────────────────────

def _det(status=DetectionStatus.OK, conf=0.9, yaw=0.0) -> DetectionResult:
    return DetectionResult(
        status=status, confidence=conf,
        head_pose=HeadPose(yaw=yaw, pitch=0.0, roll=0.0) if status != DetectionStatus.NO_FACE else None,
    )


def _metrics(ear=0.30, blink=False, blink_ms=0.0,
             state=EyeState.OPEN, fatigue=FatigueLevel.ALERT) -> EyeMetrics:
    return EyeMetrics(
        ear_left=ear, ear_right=ear, ear_avg=ear,
        ear_left_smooth=ear, ear_right_smooth=ear, ear_avg_smooth=ear,
        ear_baseline=ear, ear_deviation=0.0,
        state_left=state, state_right=state,
        blink_detected=blink, blink_duration_ms=blink_ms,
        blink_rate_30s=0.0, blink_rate_60s=0.0,
        avg_blink_duration_ms=0.0, long_blink_count=0,
        perclos_30s=0.0, perclos_60s=0.0,
        long_closure_count_60s=0, total_closure_time_60s_ms=0.0,
        fatigue_score=0.0, fatigue_level=fatigue,
    )


def _distr(face_present=True) -> DistractionResult:
    return DistractionResult(
        face_present=face_present, face_absent_ms=0.0, face_absent_alert=False,
        yaw=0.0, pitch=0.0, roll=0.0,
        is_looking_away=False, is_head_tilted=False,
        confidence=0.9 if face_present else 0.0,
        is_low_confidence=not face_present,
        distraction_score=0.0, is_distracted=False,
        reason=DistractionReason.NONE,
    )


def _cv_state() -> CVState:
    return CVState(EyeStateCV.OPEN, FatigueLevelCV.NORMAL, AttentionState.ATTENTIVE)


# ─── QualityFlags ─────────────────────────────────────────────────────────────

class TestQualityFlags(unittest.TestCase):
    def test_ok_detection_no_flags(self):
        flags = compute_quality_flags(_det(), _metrics())
        self.assertFalse(flags)

    def test_no_face_gives_no_face_flag(self):
        flags = compute_quality_flags(_det(DetectionStatus.NO_FACE, conf=0.0), None)
        self.assertIn(QualityFlag.NO_FACE, flags)

    def test_no_face_no_other_flags(self):
        flags = compute_quality_flags(_det(DetectionStatus.NO_FACE, conf=0.0), None)
        self.assertEqual(flags, frozenset({QualityFlag.NO_FACE}))

    def test_bad_angle_flag(self):
        flags = compute_quality_flags(_det(DetectionStatus.BAD_ANGLE), _metrics())
        self.assertIn(QualityFlag.BAD_ANGLE, flags)

    def test_face_too_small_flag(self):
        flags = compute_quality_flags(_det(DetectionStatus.FACE_TOO_SMALL, conf=0.1), None)
        self.assertIn(QualityFlag.FACE_TOO_SMALL, flags)

    def test_low_confidence_flag(self):
        flags = compute_quality_flags(_det(conf=0.10), _metrics())
        self.assertIn(QualityFlag.LOW_CONFIDENCE, flags)

    def test_low_ear_flag(self):
        flags = compute_quality_flags(_det(), _metrics(ear=0.10))
        self.assertIn(QualityFlag.LOW_EAR, flags)

    def test_partial_occlusion_asymmetric_ear(self):
        m = _metrics()
        # Manually build asymmetric metrics
        from dataclasses import replace
        m2 = EyeMetrics(
            ear_left=0.30, ear_right=0.30, ear_avg=0.30,
            ear_left_smooth=0.30, ear_right_smooth=0.05,  # big asymmetry
            ear_avg_smooth=0.175,
            ear_baseline=0.30, ear_deviation=0.0,
            state_left=EyeState.OPEN, state_right=EyeState.CLOSED,
            blink_detected=False, blink_duration_ms=0.0,
            blink_rate_30s=0.0, blink_rate_60s=0.0,
            avg_blink_duration_ms=0.0, long_blink_count=0,
            perclos_30s=0.0, perclos_60s=0.0,
            long_closure_count_60s=0, total_closure_time_60s_ms=0.0,
            fatigue_score=0.0, fatigue_level=FatigueLevel.ALERT,
        )
        flags = compute_quality_flags(_det(), m2)
        self.assertIn(QualityFlag.PARTIAL_OCCL, flags)

    def test_flags_to_str_ok(self):
        self.assertEqual(flags_to_str(frozenset()), "ok")

    def test_flags_to_str_sorted(self):
        flags = frozenset({QualityFlag.LOW_CONFIDENCE, QualityFlag.BAD_ANGLE})
        result = flags_to_str(flags)
        self.assertIn("bad_angle", result)
        self.assertIn("low_confidence", result)


# ─── MetricsLogger ────────────────────────────────────────────────────────────

class TestMetricsLogger(unittest.TestCase):
    def _run_logger(self, rows=3, blink=False) -> tuple[Path, Path]:
        with tempfile.TemporaryDirectory() as tmp:
            stem = str(Path(tmp) / "test_session")
            with MetricsLogger(stem) as lg:
                for i in range(rows):
                    lg.log(
                        1_000_000.0 + i * 33,
                        _det(),
                        _metrics(blink=(i == 1 and blink), blink_ms=120.0 if blink else 0.0),
                        _distr(),
                        _cv_state(),
                        frozenset(),
                    )
            csv_path   = Path(stem + ".csv")
            jsonl_path = Path(stem + "_blinks.jsonl")
            # Copy to a persistent tmp dir since TemporaryDirectory is deleted
            import shutil, tempfile as tf2
            d2 = tf2.mkdtemp()
            shutil.copy(csv_path,   d2)
            shutil.copy(jsonl_path, d2)
            return Path(d2) / csv_path.name, Path(d2) / jsonl_path.name

    def test_csv_row_count(self):
        csv_path, _ = self._run_logger(rows=5)
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 5)

    def test_csv_has_required_columns(self):
        csv_path, _ = self._run_logger()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
        for col in ("timestamp_ms", "ear_avg", "perclos_60s",
                    "fatigue_score", "attention", "quality_flags"):
            self.assertIn(col, cols)

    def test_csv_quality_flag_ok_when_no_issues(self):
        csv_path, _ = self._run_logger()
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["quality_flags"], "ok")

    def test_blink_event_logged_to_jsonl(self):
        _, jsonl_path = self._run_logger(rows=3, blink=True)
        lines = jsonl_path.read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertIn("duration_ms", rec)
        self.assertIn("is_long", rec)

    def test_no_blink_jsonl_empty(self):
        _, jsonl_path = self._run_logger(rows=3, blink=False)
        self.assertEqual(jsonl_path.read_text().strip(), "")

    def test_none_metrics_logged_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            stem = str(Path(tmp) / "s")
            with MetricsLogger(stem) as lg:
                lg.log(1_000_000.0, _det(DetectionStatus.NO_FACE, 0.0),
                       None, _distr(False), _cv_state(), frozenset())
            with open(stem + ".csv") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["ear_avg"], "")


# ─── MetricsPlotter ──────────────────────────────────────────────────────────

class TestMetricsPlotter(unittest.TestCase):
    def _plotter(self):
        return MetricsPlotter(window_seconds=10.0, show_window=False)

    def test_draw_without_data_returns_array(self):
        p = self._plotter()
        canvas = p.draw(1_000_000.0)
        self.assertIsInstance(canvas, np.ndarray)
        self.assertEqual(len(canvas.shape), 3)   # H×W×3

    def test_draw_with_data_no_crash(self):
        p = self._plotter()
        base = 1_000_000.0
        for i in range(50):
            p.update(base + i * 200, _metrics(), _distr())
        canvas = p.draw(base + 50 * 200)
        self.assertIsInstance(canvas, np.ndarray)

    def test_old_samples_pruned(self):
        p = self._plotter()   # 10s window
        base = 1_000_000.0
        for i in range(20):
            p.update(base + i * 200, _metrics(), _distr())   # 4s of data
        # Jump 15s forward — all old samples should be pruned
        p.update(base + 15_000, _metrics(), _distr())
        for buf in p._data:
            for t, _ in buf:
                self.assertGreaterEqual(t, base + 15_000 - 10_000)

    def test_canvas_size_correct(self):
        p = self._plotter()
        canvas = p.draw(1_000_000.0)
        h, w = canvas.shape[:2]
        self.assertEqual(w, 600)
        self.assertEqual(h, 180)


if __name__ == "__main__":
    unittest.main()
