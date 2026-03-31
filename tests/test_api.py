"""Tests for Stage 8: API / integration interface."""

import asyncio
import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.landmarks.result import DetectionResult, DetectionStatus, HeadPose
from app.metrics.ear import EyeState
from app.metrics.result import EyeMetrics, FatigueLevel
from app.state.distraction import DistractionResult, DistractionReason
from app.state.cv_state import CVState, EyeStateCV, FatigueLevelCV, AttentionState
from app.api.serializer import serialize_frame, serialize_event, _metrics_dict
from app.api.server import WebSocketServer


# ─── helpers ─────────────────────────────────────────────────────────────────

def _metrics() -> EyeMetrics:
    return EyeMetrics(
        ear_left=0.30, ear_right=0.30, ear_avg=0.30,
        ear_left_smooth=0.30, ear_right_smooth=0.30, ear_avg_smooth=0.30,
        ear_baseline=0.30, ear_deviation=0.0,
        state_left=EyeState.OPEN, state_right=EyeState.OPEN,
        blink_detected=False, blink_duration_ms=0.0,
        blink_rate_30s=12.0, blink_rate_60s=14.0,
        avg_blink_duration_ms=130.0, long_blink_count=0,
        perclos_30s=0.02, perclos_60s=0.01,
        long_closure_count_60s=0, total_closure_time_60s_ms=0.0,
        fatigue_score=0.05, fatigue_level=FatigueLevel.ALERT,
    )


def _distr() -> DistractionResult:
    return DistractionResult(
        face_present=True, face_absent_ms=0.0, face_absent_alert=False,
        yaw=2.5, pitch=-3.0, roll=0.5,
        is_looking_away=False, is_head_tilted=False,
        confidence=0.9, is_low_confidence=False,
        distraction_score=0.0, is_distracted=False,
        reason=DistractionReason.NONE,
    )


def _cv_state() -> CVState:
    return CVState(
        eye_state=EyeStateCV.OPEN,
        fatigue_level=FatigueLevelCV.NORMAL,
        attention=AttentionState.ATTENTIVE,
    )


# ─── serializer ──────────────────────────────────────────────────────────────

class TestSerializer(unittest.TestCase):
    def _parse(self, metrics=None) -> dict:
        raw = serialize_frame(1_000_000.0, _cv_state(),
                              metrics if metrics is not None else _metrics(), _distr())
        return json.loads(raw)

    def test_type_is_frame(self):
        self.assertEqual(self._parse()["type"], "frame")

    def test_cv_states_present(self):
        d = self._parse()
        self.assertEqual(d["eye_state"],     "open")
        self.assertEqual(d["fatigue_level"], "normal")
        self.assertEqual(d["attention"],     "attentive")

    def test_metrics_fields_present(self):
        d = self._parse()
        m = d["metrics"]
        self.assertIn("ear_avg",         m)
        self.assertIn("blink_detected",  m)
        self.assertIn("perclos_60s",     m)
        self.assertIn("fatigue_score",   m)

    def test_distraction_fields_present(self):
        d = self._parse()
        dist = d["distraction"]
        self.assertIn("face_present",       dist)
        self.assertIn("yaw",                dist)
        self.assertIn("is_distracted",      dist)
        self.assertIn("distraction_score",  dist)
        self.assertIn("reason",             dist)

    def test_metrics_none_serializes_as_null(self):
        raw = serialize_frame(1_000_000.0, _cv_state(), None, _distr())
        d = json.loads(raw)
        self.assertIsNone(d["metrics"])

    def test_output_is_valid_json(self):
        raw = serialize_frame(1_000_000.0, _cv_state(), _metrics(), _distr())
        self.assertIsInstance(json.loads(raw), dict)

    def test_serialize_event(self):
        raw = serialize_event("session_started")
        d = json.loads(raw)
        self.assertEqual(d["type"], "session_started")

    def test_serialize_event_with_kwargs(self):
        raw = serialize_event("calibration_done", profile_valid=True)
        d = json.loads(raw)
        self.assertEqual(d["type"], "calibration_done")
        self.assertTrue(d["profile_valid"])

    def test_timestamp_in_output(self):
        d = self._parse()
        self.assertAlmostEqual(d["timestamp_ms"], 1_000_000.0, places=0)


# ─── WebSocket server lifecycle ───────────────────────────────────────────────

class TestWebSocketServer(unittest.TestCase):
    def test_start_and_stop_no_crash(self):
        srv = WebSocketServer(port=8790)
        srv.start()
        time.sleep(0.15)   # let the server bind
        srv.stop()

    def test_push_before_start_does_not_crash(self):
        """push() with no loop should silently do nothing."""
        srv = WebSocketServer(port=8791)
        srv.push('{"type":"frame"}')   # no crash

    def test_push_event_before_start_does_not_crash(self):
        srv = WebSocketServer(port=8792)
        srv.push_event("session_started")

    def test_client_count_zero_initially(self):
        srv = WebSocketServer(port=8793)
        self.assertEqual(srv.client_count, 0)

    def test_on_command_called_for_known_cmd(self):
        """on_command callback must be invoked when a client sends a command."""
        received = []
        srv = WebSocketServer(port=8794, on_command=received.append)
        srv.start()
        time.sleep(0.15)

        async def _client():
            import websockets as ws
            async with ws.connect("ws://localhost:8794") as conn:
                await conn.send('{"cmd":"start_calibration"}')
                await conn.recv()   # ack

        asyncio.run(_client())
        time.sleep(0.05)
        srv.stop()
        self.assertIn("start_calibration", received)

    def test_get_current_metrics_returns_last_frame(self):
        srv = WebSocketServer(port=8795)
        srv.start()
        time.sleep(0.15)

        frame_json = serialize_frame(1_000_000.0, _cv_state(), _metrics(), _distr())
        srv.push(frame_json)
        time.sleep(0.05)

        async def _client():
            import websockets as ws
            async with ws.connect("ws://localhost:8795") as conn:
                await conn.send('{"cmd":"get_current_metrics"}')
                reply = await conn.recv()
                return json.loads(reply)

        result = asyncio.run(_client())
        srv.stop()
        self.assertEqual(result["type"], "frame")

    def test_broadcast_reaches_client(self):
        srv = WebSocketServer(port=8796)
        srv.start()
        time.sleep(0.15)

        messages = []

        async def _client():
            import websockets as ws
            async with ws.connect("ws://localhost:8796") as conn:
                # Give server time to see the connection, then push
                await asyncio.sleep(0.05)
                srv.push('{"type":"frame","eye_state":"open"}')
                msg = await asyncio.wait_for(conn.recv(), timeout=1.0)
                messages.append(json.loads(msg))

        asyncio.run(_client())
        srv.stop()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["eye_state"], "open")


if __name__ == "__main__":
    unittest.main()
