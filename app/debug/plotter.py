"""
MetricsPlotter — Stage 9.

Renders a rolling time-series graph of key metrics using only OpenCV.
Displayed in a separate "Metrics" window or embedded in the main frame.

Tracks (last 60 s):
  EAR avg (blue)       — scale 0–0.5
  PERCLOS 30s (red)    — scale 0–0.4
  Fatigue score (orange) — scale 0–1
"""

from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

from app.metrics.result import EyeMetrics


# ─── constants ────────────────────────────────────────────────────────────────

_W, _H   = 600, 180      # canvas size
_PAD     = 40             # left/bottom padding for axes
_WINDOW  = 60_000.0       # ms to keep (60 s)

_SERIES = [
    # (label,       colour_bgr,     y_max, key_fn)
    ("EAR",         (220, 180,  60), 0.50, lambda m, _d: m.ear_avg_smooth   if m else None),
    ("PERCLOS 30s", ( 60,  60, 220), 0.40, lambda m, _d: m.perclos_30s      if m else None),
    ("Fatigue",     ( 40, 180, 220), 1.00, lambda m, _d: m.fatigue_score    if m else None),
]


class MetricsPlotter:
    """
    Call update() every frame, then draw() to render the graph.

    Parameters
    ----------
    window_seconds : rolling window length shown in graph (default 60 s)
    show_window    : if True, display in a separate OpenCV window
    """

    def __init__(self, window_seconds: float = 60.0, show_window: bool = True):
        self._window_ms  = window_seconds * 1_000.0
        self._show_window = show_window
        # deque of (timestamp_ms, value) per series
        self._data: list[deque[Tuple[float, float]]] = [
            deque() for _ in _SERIES
        ]

    # ─── public API ──────────────────────────────────────────────────────────

    def update(
        self,
        timestamp_ms: float,
        metrics:      Optional[EyeMetrics],
        distraction,          # DistractionResult, typed loosely to avoid circular
    ):
        """Record one frame of data."""
        for i, (_, _, _, key_fn) in enumerate(_SERIES):
            value = key_fn(metrics, distraction)
            if value is not None:
                self._data[i].append((timestamp_ms, float(value)))

        # Prune old samples
        cutoff = timestamp_ms - self._window_ms
        for buf in self._data:
            while buf and buf[0][0] < cutoff:
                buf.popleft()

    def draw(self, timestamp_ms: float) -> np.ndarray:
        """Render graph and optionally show in its own window. Returns canvas."""
        canvas = self._render(timestamp_ms)
        if self._show_window:
            cv2.imshow("Metrics", canvas)
        return canvas

    # ─── private ─────────────────────────────────────────────────────────────

    def _render(self, now_ms: float) -> np.ndarray:
        canvas = np.full((_H, _W, 3), 20, dtype=np.uint8)

        plot_w = _W - _PAD - 10
        plot_h = _H - _PAD - 10
        plot_x = _PAD
        plot_y = 10

        # Grid lines
        for frac in (0.25, 0.5, 0.75, 1.0):
            y = plot_y + int(plot_h * (1.0 - frac))
            cv2.line(canvas, (plot_x, y), (plot_x + plot_w, y), (45, 45, 45), 1)

        # Axes
        cv2.line(canvas, (plot_x, plot_y), (plot_x, plot_y + plot_h),
                 (100, 100, 100), 1)
        cv2.line(canvas, (plot_x, plot_y + plot_h),
                 (plot_x + plot_w, plot_y + plot_h), (100, 100, 100), 1)

        t_start = now_ms - self._window_ms

        # Series lines + legend
        for i, (label, colour, y_max, _) in enumerate(_SERIES):
            buf = self._data[i]
            if len(buf) < 2:
                continue

            pts = []
            for t, v in buf:
                x = plot_x + int((t - t_start) / self._window_ms * plot_w)
                y = plot_y + plot_h - int(min(v / y_max, 1.0) * plot_h)
                pts.append((x, y))

            for j in range(1, len(pts)):
                cv2.line(canvas, pts[j - 1], pts[j], colour, 1, cv2.LINE_AA)

            # Legend row
            lx = plot_x + 4 + i * 170
            ly = plot_y + plot_h + 20
            # Current value
            cur = buf[-1][1] if buf else 0.0
            cv2.putText(canvas, f"{label}: {cur:.3f}",
                        (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        colour, 1, cv2.LINE_AA)

        # Time label
        cv2.putText(canvas, f"-{int(self._window_ms/1000)}s",
                    (plot_x, plot_y + plot_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1)
        cv2.putText(canvas, "now",
                    (plot_x + plot_w - 20, plot_y + plot_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1)

        return canvas
