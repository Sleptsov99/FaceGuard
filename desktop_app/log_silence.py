"""
Reduce TensorFlow / MediaPipe / oneDNN console noise.

1) Environment variables (call ``apply_log_silence()`` as early as possible).
2) Optional stderr filter while the CV worker runs — MediaPipe/TFLite often
   bypass Python logging and write directly to stderr (W0000 / INFO lines).
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Iterator, TextIO


def apply_log_silence() -> None:
    """Set env vars so C++ loggers stay quieter (not all builds respect them)."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("GLOG_logtostderr", "0")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("ABSL_LOGGING_MIN_LOG_LEVEL", "2")


# Substrings of known noisy MediaPipe / TFLite startup lines (stderr only).
_MEDIAPIPE_STDERR_SUPPRESS = (
    "face_landmarker_graph.cc",
    "inference_feedback_manager.cc",
    "TensorFlow Lite XNNPACK delegate",
    "FaceBlendshapesGraph acceleration",
    "Created TensorFlow Lite",
)


class _FilteredStderr:
    def __init__(self, real: TextIO) -> None:
        self._real = real

    def write(self, s: str) -> int:
        if not isinstance(s, str):
            s = str(s)
        if any(p in s for p in _MEDIAPIPE_STDERR_SUPPRESS):
            return len(s)
        return self._real.write(s)

    def flush(self) -> None:
        self._real.flush()

    def __getattr__(self, name: str):
        return getattr(self._real, name)


@contextlib.contextmanager
def mediapipe_stderr_filter() -> Iterator[None]:
    """
    Temporarily wrap ``sys.stderr`` to drop known MediaPipe/TFLite spam.

    Use inside the engine worker thread around session creation and the main loop.
    """
    prev = sys.stderr
    sys.stderr = _FilteredStderr(prev)
    try:
        yield
    finally:
        sys.stderr = prev
