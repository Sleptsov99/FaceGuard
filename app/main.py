"""
CV Engine - Main entry point

Usage:
    python app/main.py --source camera
    python app/main.py --source camera --mode ws
    python app/main.py --source camera --mode both
    python app/main.py --source video --path <video_file>
    python app/main.py --source camera --log session1 --plot
    python app/main.py --source camera --record out.avi

Modes:
    window  — local debug window only (default)
    ws      — headless WebSocket server on ws://localhost:8765
    both    — window + WebSocket simultaneously

Debug flags:
    --log <stem>     write CSV + blink JSONL (e.g. --log session1)
    --plot           show rolling metrics graph in a separate window
    --record <path>  save video with overlay to file (e.g. --record out.avi)

Terminal commands (type + Enter):
    c  — start / restart calibration session
    q  — quit
"""

import argparse
import cv2
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.camera.capture import CameraCapture
from app.landmarks.detector import LandmarkDetector
from app.metrics.calculator import MetricsCalculator
from app.state.distraction import DistractionTracker
from app.state.calibration import CalibrationSession, CalibrationState
from app.state.cv_state import CVStateEstimator
from app.api.server import WebSocketServer
from app.api.serializer import serialize_frame
from app.debug.quality import compute_quality_flags
from app.debug.logger import MetricsLogger
from app.debug.plotter import MetricsPlotter


def _start_stdin_reader() -> queue.SimpleQueue:
    q: queue.SimpleQueue = queue.SimpleQueue()

    def _reader():
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                ch = line.strip().lower()
                if ch:
                    q.put(ch[0])
            except Exception:
                break

    threading.Thread(target=_reader, daemon=True).start()
    return q


def run(
    capture:    CameraCapture,
    detector:   LandmarkDetector,
    mode:       str,
    ws_port:    int,
    log_stem:   Optional[str],
    do_plot:    bool,
    record_path: Optional[str],
):
    calculator  = MetricsCalculator()
    distraction = DistractionTracker()
    calibration = CalibrationSession(duration_seconds=45.0)
    cv_state    = CVStateEstimator()
    stdin_q     = _start_stdin_reader()

    show_window = mode in ("window", "both")

    # ── optional debug components ─────────────────────────────────────────────
    logger:   Optional[MetricsLogger]  = MetricsLogger(log_stem) if log_stem else None
    plotter:  Optional[MetricsPlotter] = MetricsPlotter(show_window=True) if do_plot else None
    writer:   Optional[cv2.VideoWriter] = None   # initialised on first frame

    # ── WebSocket server ──────────────────────────────────────────────────────
    ws: Optional[WebSocketServer] = None
    if mode in ("ws", "both"):
        def _on_ws_command(cmd: str):
            if cmd == "start_calibration":
                stdin_q.put("c")
            elif cmd == "stop_session":
                stdin_q.put("q")

        ws = WebSocketServer(port=ws_port, on_command=_on_ws_command)
        ws.start()

    print("CV Engine running.")
    print("  c + Enter  — start calibration")
    print("  q + Enter  — quit")
    if ws:
        print(f"  WebSocket   — ws://localhost:{ws_port}")
    if logger:
        pass   # MetricsLogger already prints file paths in __init__
    if record_path:
        print(f"  Recording   — {record_path}")

    if ws:
        ws.push_event("session_started")

    quit_flag = False
    while capture.is_opened() and not quit_flag:
        frame = capture.read()
        if frame is None:
            break

        ts = time.time() * 1_000.0

        detection = detector.detect(frame)
        metrics   = calculator.update(detection, timestamp_ms=ts)
        distr     = distraction.update(detection, timestamp_ms=ts)
        state     = cv_state.estimate(metrics, distr)
        quality   = compute_quality_flags(detection, metrics)

        # Feed calibration if running
        if calibration.state == CalibrationState.RUNNING:
            cal_state = calibration.update(detection, metrics, timestamp_ms=ts)
            if cal_state == CalibrationState.DONE:
                profile = calibration.finish()
                if profile:
                    calculator.apply_calibration(profile)
                    distraction.apply_calibration(profile)
                    print("Calibration done — personal thresholds applied")
                    if ws:
                        ws.push_event("calibration_done")
            elif cal_state == CalibrationState.FAILED:
                print("Calibration failed — not enough valid frames")
                if ws:
                    ws.push_event("calibration_failed")

        # ── logging ───────────────────────────────────────────────────────────
        if logger:
            logger.log(ts, detection, metrics, distr, state, quality)

        # ── plotter ───────────────────────────────────────────────────────────
        if plotter:
            plotter.update(ts, metrics, distr)
            plotter.draw(ts)

        # ── WebSocket broadcast ───────────────────────────────────────────────
        if ws:
            ws.push(serialize_frame(ts, state, metrics, distr))

        # ── window rendering ──────────────────────────────────────────────────
        if show_window or record_path:
            detector.draw(frame, detection)
            calculator.draw(frame, metrics)
            distraction.draw(frame, distr)
            cv_state.draw(frame, state)
            _draw_quality(frame, quality)
            calibration.draw(frame, timestamp_ms=ts)

            if show_window:
                cv2.imshow("CV Engine", frame)

            # ── video recording ───────────────────────────────────────────────
            if record_path:
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    writer = cv2.VideoWriter(record_path, fourcc, 30.0, (w, h))
                writer.write(frame)

        cv2.waitKey(1)

        # ── terminal commands ─────────────────────────────────────────────────
        while not stdin_q.empty():
            ch = stdin_q.get()
            if ch == 'q':
                quit_flag = True
            elif ch == 'c':
                print("Calibration started — look at the screen normally")
                calibration.start(timestamp_ms=ts)
                if ws:
                    ws.push_event("calibration_started")

    # ── cleanup ───────────────────────────────────────────────────────────────
    if ws:
        ws.push_event("session_stopped")
        ws.stop()
    if logger:
        logger.close()
    if writer:
        writer.release()

    capture.release()
    detector.release()
    cv2.destroyAllWindows()


def _draw_quality(frame, quality):
    """Draw quality flags in bottom-left if any issues detected."""
    from app.debug.quality import QualityFlag
    if not quality or quality == frozenset():
        return
    bad = {QualityFlag.NO_FACE, QualityFlag.LOW_CONFIDENCE,
           QualityFlag.BAD_ANGLE, QualityFlag.FACE_TOO_SMALL,
           QualityFlag.LOW_EAR, QualityFlag.PARTIAL_OCCL}
    issues = quality & bad
    if not issues:
        return
    h = frame.shape[0]
    label = " | ".join(sorted(f.value for f in issues))
    cv2.putText(frame, f"[Q] {label}",
                (10, h - 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 140, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="CV Engine")
    parser.add_argument("--source",   choices=["camera", "video"], default="camera")
    parser.add_argument("--path",     type=str, default=None,
                        help="Path to video file (--source video)")
    parser.add_argument("--mode",     choices=["window", "ws", "both"], default="window",
                        help="Output mode (default: window)")
    parser.add_argument("--ws-port",  type=int, default=8765)
    parser.add_argument("--log",      type=str, default=None, metavar="STEM",
                        help="Enable CSV/JSONL logging, e.g. --log session1")
    parser.add_argument("--plot",     action="store_true",
                        help="Show rolling metrics graph window")
    parser.add_argument("--record",   type=str, default=None, metavar="FILE",
                        help="Save video with overlay, e.g. --record out.avi")
    args = parser.parse_args()

    if args.source == "video":
        if not args.path:
            print("Error: --path is required for video mode")
            sys.exit(1)
        source = args.path
    else:
        source = 0

    run(
        CameraCapture(source=source),
        LandmarkDetector(),
        mode=args.mode,
        ws_port=args.ws_port,
        log_stem=args.log,
        do_plot=args.plot,
        record_path=args.record,
    )


if __name__ == "__main__":
    main()
