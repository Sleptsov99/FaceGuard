"""
CV Engine - Main entry point

Usage:
    python app/main.py --source camera
    python app/main.py --source video --path <video_file>
"""

import argparse
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.camera.capture import CameraCapture
from app.landmarks.detector import LandmarkDetector
from app.metrics.calculator import MetricsCalculator
from app.state.distraction import DistractionTracker


def run(capture: CameraCapture, detector: LandmarkDetector):
    calculator  = MetricsCalculator()
    distraction = DistractionTracker()

    while capture.is_opened():
        frame = capture.read()
        if frame is None:
            break

        detection   = detector.detect(frame)
        metrics     = calculator.update(detection)     # None when not OK
        distr       = distraction.update(detection)

        detector.draw(frame, detection)
        calculator.draw(frame, metrics)
        distraction.draw(frame, distr)

        cv2.imshow("CV Engine", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    detector.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="CV Engine - Facial Analysis")
    parser.add_argument("--source", choices=["camera", "video"], default="camera")
    parser.add_argument("--path", type=str, default=None,
                        help="Path to video file (required for video mode)")
    args = parser.parse_args()

    if args.source == "video":
        if not args.path:
            print("Error: --path is required for video mode")
            sys.exit(1)
        source = args.path
    else:
        source = 0

    run(CameraCapture(source=source), LandmarkDetector())


if __name__ == "__main__":
    main()
