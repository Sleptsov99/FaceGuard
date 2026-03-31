"""
Facial landmark detection — Stage 1.

Uses **MediaPipe Tasks** ``FaceLandmarker`` (``mediapipe>=0.10.30``). The legacy
``mp.solutions.face_mesh`` API was removed from PyPI builds for recent Python
(e.g. 3.13), where only 0.10.30+ wheels exist. Landmark indices match the
478-point refined face mesh (see ``indices.py``).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from app.landmarks.indices import (
    RIGHT_EYE_CONTOUR,
    LEFT_EYE_CONTOUR,
    RIGHT_EYE_EAR,
    LEFT_EYE_EAR,
    POSE_KEYPOINTS,
    FACE_3D_MODEL,
    FACE_OVAL,
)
from app.landmarks.result import (
    DetectionResult,
    DetectionStatus,
    EyeLandmarks,
    HeadPose,
)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# ─── draw colours ────────────────────────────────────────────────────────────
_STATUS_COLOUR = {
    DetectionStatus.OK:             (0, 255,   0),
    DetectionStatus.NO_FACE:        (128, 128, 128),
    DetectionStatus.FACE_TOO_SMALL: (0, 165, 255),
    DetectionStatus.BAD_ANGLE:      (0, 165, 255),
    DetectionStatus.MULTIPLE_FACES: (0,   0, 255),
}
_STATUS_LABEL = {
    DetectionStatus.OK:             "OK",
    DetectionStatus.NO_FACE:        "No face",
    DetectionStatus.FACE_TOO_SMALL: "Face too small",
    DetectionStatus.BAD_ANGLE:      "Bad angle",
    DetectionStatus.MULTIPLE_FACES: "Multiple faces",
}


def _ensure_face_landmarker_model() -> str:
    """Download bundled task model once into app/landmarks/data/."""
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "face_landmarker.task"
    if not path.exists():
        urllib.request.urlretrieve(_MODEL_URL, path)
    return str(path)


class LandmarkDetector:
    """Detects facial landmarks and classifies detection quality."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_face_size_ratio: float = 0.04,   # face bbox area / frame area
        max_yaw_degrees: float = 30.0,
        max_pitch_degrees: float = 30.0,
    ):
        self.min_face_size_ratio = min_face_size_ratio
        self.max_yaw_degrees = max_yaw_degrees
        self.max_pitch_degrees = max_pitch_degrees

        model_path = _ensure_face_landmarker_model()
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=2,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._video_ts_ms = 0

    # ─── public API ──────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Process one BGR frame and return a DetectionResult."""
        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._video_ts_ms += 33
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        mp_result = self._landmarker.detect_for_video(mp_image, self._video_ts_ms)

        faces = mp_result.face_landmarks
        if not faces:
            return DetectionResult(status=DetectionStatus.NO_FACE)

        # ── multiple faces ────────────────────────────────────────────────
        if len(faces) > 1:
            lm = faces[0]
            bbox = self._face_bbox(lm, h, w)
            return DetectionResult(
                status=DetectionStatus.MULTIPLE_FACES,
                confidence=self._confidence(bbox, h, w),
                face_bbox=bbox,
                right_eye=self._eye(lm, RIGHT_EYE_CONTOUR, RIGHT_EYE_EAR, h, w),
                left_eye=self._eye(lm, LEFT_EYE_CONTOUR, LEFT_EYE_EAR, h, w),
                head_pose=self._head_pose(lm, h, w),
                raw_landmarks=None,
            )

        lm = faces[0]
        bbox = self._face_bbox(lm, h, w)

        # ── face too small ────────────────────────────────────────────────
        bx, by, bw, bh = bbox
        if (bw * bh) / (w * h) < self.min_face_size_ratio:
            return DetectionResult(
                status=DetectionStatus.FACE_TOO_SMALL,
                confidence=self._confidence(bbox, h, w),
                face_bbox=bbox,
                raw_landmarks=None,
            )

        head_pose = self._head_pose(lm, h, w)
        right_eye = self._eye(lm, RIGHT_EYE_CONTOUR, RIGHT_EYE_EAR, h, w)
        left_eye = self._eye(lm, LEFT_EYE_CONTOUR, LEFT_EYE_EAR, h, w)
        confidence = self._confidence(bbox, h, w)

        # ── bad angle ─────────────────────────────────────────────────────
        if not head_pose.is_frontal(self.max_yaw_degrees, self.max_pitch_degrees):
            return DetectionResult(
                status=DetectionStatus.BAD_ANGLE,
                confidence=confidence,
                face_bbox=bbox,
                right_eye=right_eye,
                left_eye=left_eye,
                head_pose=head_pose,
                raw_landmarks=None,
            )

        return DetectionResult(
            status=DetectionStatus.OK,
            confidence=confidence,
            face_bbox=bbox,
            right_eye=right_eye,
            left_eye=left_eye,
            head_pose=head_pose,
            raw_landmarks=None,
        )

    def draw(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw debug overlay onto frame (in-place). Returns frame."""
        colour = _STATUS_COLOUR[result.status]
        label = _STATUS_LABEL[result.status]

        if result.face_bbox:
            x, y, fw, fh = result.face_bbox
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), colour, 2)

        if result.right_eye:
            self._draw_eye(frame, result.right_eye, colour)
        if result.left_eye:
            self._draw_eye(frame, result.left_eye, colour)

        cv2.putText(
            frame,
            f"{label}  conf:{result.confidence:.2f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour,
            2,
            cv2.LINE_AA,
        )

        if result.head_pose:
            pose = result.head_pose
            cv2.putText(
                frame,
                f"Y:{pose.yaw:+.1f}  P:{pose.pitch:+.1f}  R:{pose.roll:+.1f}",
                (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        return frame

    def release(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    # ─── private helpers ─────────────────────────────────────────────────────

    def _face_bbox(
        self,
        lm: Sequence,
        h: int,
        w: int,
    ) -> Tuple[int, int, int, int]:
        xs = [lm[i].x * w for i in FACE_OVAL]
        ys = [lm[i].y * h for i in FACE_OVAL]
        x0, x1 = int(min(xs)), int(max(xs))
        y0, y1 = int(min(ys)), int(max(ys))
        return (x0, y0, x1 - x0, y1 - y0)

    def _eye(
        self,
        lm: Sequence,
        contour_idx: List[int],
        ear_idx: List[int],
        h: int,
        w: int,
    ) -> EyeLandmarks:
        contour = np.array(
            [[lm[i].x * w, lm[i].y * h] for i in contour_idx],
            dtype=np.float32,
        )
        ear_pts = np.array(
            [[lm[i].x * w, lm[i].y * h] for i in ear_idx],
            dtype=np.float32,
        )
        return EyeLandmarks(contour=contour, ear_points=ear_pts)

    def _head_pose(self, lm: Sequence, h: int, w: int) -> HeadPose:
        image_pts = np.array(
            [[lm[i].x * w, lm[i].y * h] for i in POSE_KEYPOINTS],
            dtype=np.float64,
        )
        focal = float(w)
        cam_matrix = np.array(
            [
                [focal, 0, w / 2],
                [0, focal, h / 2],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        ok, rvec, _ = cv2.solvePnP(
            FACE_3D_MODEL,
            image_pts,
            cam_matrix,
            np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return HeadPose(yaw=0.0, pitch=0.0, roll=0.0)

        rot, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
        if sy > 1e-6:
            pitch = np.arctan2(rot[2, 1], rot[2, 2])
            yaw = np.arctan2(-rot[2, 0], sy)
            roll = np.arctan2(rot[1, 0], rot[0, 0])
        else:
            pitch = np.arctan2(-rot[1, 2], rot[1, 1])
            yaw = np.arctan2(-rot[2, 0], sy)
            roll = 0.0

        return HeadPose(
            yaw=float(np.degrees(yaw)),
            pitch=float(np.degrees(pitch)),
            roll=float(np.degrees(roll)),
        )

    def _confidence(self, bbox: Tuple[int, int, int, int], h: int, w: int) -> float:
        _, _, bw, bh = bbox
        ratio = (bw * bh) / (w * h)
        return float(np.clip(ratio / 0.30, 0.05, 1.0))

    @staticmethod
    def _draw_eye(frame: np.ndarray, eye: EyeLandmarks, colour: tuple):
        pts = eye.contour.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=1)
