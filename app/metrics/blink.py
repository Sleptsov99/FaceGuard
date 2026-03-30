"""Blink detection and statistics via a sliding-window approach.

Stage 3 addition: consecutive-frame debounce.
  min_frames_closed — how many consecutive below-threshold frames must be seen
                      before a blink is considered to have started.
  min_frames_open   — how many consecutive above-threshold frames must be seen
                      before a blink is considered to have ended.

When the input EAR is already EMA-smoothed (recommended), single noisy frames
won't even reach the threshold, so the defaults of 1 are safe.
Set to 2 for an additional explicit guard on top of EMA smoothing.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BlinkEvent:
    start_ms: float
    end_ms: float
    duration_ms: float


@dataclass(frozen=True)
class BlinkStats:
    blink_count_30s: int
    blink_count_60s: int
    blink_rate_60s: float
    avg_blink_duration_ms: float
    long_blink_count: int


class BlinkDetector:
    """
    Stateful per-frame blink detector with consecutive-frame debounce.

    Recommended usage: pass EMA-smoothed EAR (from EarSmoother) rather than
    raw per-frame EAR to eliminate single-frame noise before it reaches here.
    """

    def __init__(
        self,
        ear_threshold: float          = 0.20,
        min_blink_duration_ms: float  = 50.0,
        max_blink_duration_ms: float  = 800.0,
        long_blink_threshold_ms: float = 400.0,
        min_frames_closed: int        = 1,   # debounce: frames below threshold to start
        min_frames_open: int          = 1,   # debounce: frames above threshold to end
    ):
        self.ear_threshold            = ear_threshold
        self.min_blink_duration_ms    = min_blink_duration_ms
        self.max_blink_duration_ms    = max_blink_duration_ms
        self.long_blink_threshold_ms  = long_blink_threshold_ms
        self.min_frames_closed        = max(1, min_frames_closed)
        self.min_frames_open          = max(1, min_frames_open)

        self._in_blink: bool                  = False
        self._blink_start_ms: Optional[float] = None
        self._waiting_for_open: bool          = False

        # Consecutive-frame counters + run start timestamps
        self._below_streak: int               = 0
        self._below_start_ms: Optional[float] = None
        self._above_streak: int               = 0
        self._above_start_ms: Optional[float] = None

        self._completed: deque[BlinkEvent] = deque()

    # ─── public ──────────────────────────────────────────────────────────────

    def update(
        self,
        ear_avg: float,
        timestamp_ms: Optional[float] = None,
    ) -> Optional[BlinkEvent]:
        """Process one frame. Returns BlinkEvent when a blink completes."""
        if timestamp_ms is None:
            timestamp_ms = time.time() * 1_000.0

        # Guard: cancel blinks that ran too long (tracking loss)
        if self._in_blink and self._blink_start_ms is not None:
            if timestamp_ms - self._blink_start_ms > self.max_blink_duration_ms:
                self._reset_streaks()
                self._in_blink = False
                self._blink_start_ms = None
                self._waiting_for_open = True

        # After timeout: wait for sustained open before resuming detection
        if self._waiting_for_open:
            if ear_avg >= self.ear_threshold:
                self._above_streak += 1
                if self._above_streak >= self.min_frames_open:
                    self._waiting_for_open = False
                    self._reset_streaks()
            else:
                self._above_streak = 0
            return None

        event: Optional[BlinkEvent] = None

        if ear_avg < self.ear_threshold:
            # ── accumulate below-threshold streak ─────────────────────────────
            if self._below_streak == 0:
                self._below_start_ms = timestamp_ms
            self._below_streak += 1
            self._above_streak = 0
            self._above_start_ms = None

            if not self._in_blink and self._below_streak >= self.min_frames_closed:
                self._in_blink = True
                self._blink_start_ms = self._below_start_ms  # use run start, not now
        else:
            # ── accumulate above-threshold streak ─────────────────────────────
            if self._above_streak == 0:
                self._above_start_ms = timestamp_ms
            self._above_streak += 1
            self._below_streak = 0
            self._below_start_ms = None

            if self._in_blink and self._above_streak >= self.min_frames_open:
                self._in_blink = False
                end_ms   = self._above_start_ms   # use run start, not current frame
                duration = end_ms - self._blink_start_ms
                self._blink_start_ms = None
                self._reset_streaks()

                if self.min_blink_duration_ms <= duration <= self.max_blink_duration_ms:
                    event = BlinkEvent(
                        start_ms=end_ms - duration,
                        end_ms=end_ms,
                        duration_ms=duration,
                    )
                    self._completed.append(event)
                    self._prune(timestamp_ms)

        return event

    def get_stats(self, timestamp_ms: Optional[float] = None) -> BlinkStats:
        if timestamp_ms is None:
            timestamp_ms = time.time() * 1_000.0

        w30 = [b for b in self._completed if b.end_ms >= timestamp_ms - 30_000]
        w60 = [b for b in self._completed if b.end_ms >= timestamp_ms - 60_000]

        durations_60 = [b.duration_ms for b in w60]
        avg_dur = sum(durations_60) / len(durations_60) if durations_60 else 0.0
        long    = sum(1 for d in durations_60 if d >= self.long_blink_threshold_ms)

        return BlinkStats(
            blink_count_30s=len(w30),
            blink_count_60s=len(w60),
            blink_rate_60s=float(len(w60)),
            avg_blink_duration_ms=avg_dur,
            long_blink_count=long,
        )

    def reset(self):
        self._in_blink = False
        self._blink_start_ms = None
        self._waiting_for_open = False
        self._reset_streaks()
        self._completed.clear()

    # ─── private ─────────────────────────────────────────────────────────────

    def _reset_streaks(self):
        self._below_streak   = 0
        self._below_start_ms = None
        self._above_streak   = 0
        self._above_start_ms = None

    def _prune(self, timestamp_ms: float):
        cutoff = timestamp_ms - 120_000
        while self._completed and self._completed[0].end_ms < cutoff:
            self._completed.popleft()
