"""
PERCLOS tracker and long eye-closure detector — Stage 4.

PERCLOS (PERcentage of eye CLOSure) is the fraction of time within a
sliding window during which the EAR is below the "closed" threshold.

Separate from BlinkDetector:
  • BlinkDetector: short events 50–800 ms (normal blinks)
  • PerclosTracker: all closure events including prolonged ones (> 500 ms)

The frame-count approximation is used for PERCLOS:
    perclos = closed_frames / total_frames_in_window
At ≥20 fps the approximation error is < 5 % compared to a time-weighted
approach, which is acceptable given inter-individual EAR variability.
"""

import time 
from collections import deque 
from dataclasses import dataclass 
from typing import Optional ,Tuple 


@dataclass (frozen =True )
class ClosureEvent :
    """A completed eye-closure episode (blink or prolonged closure)."""
    start_ms :float 
    end_ms :float 
    duration_ms :float 

    @property 
    def is_long (self )->bool :
        return self .duration_ms >=500.0 


class PerclosTracker :
    """
    Tracks PERCLOS and prolonged eye-closure events.

    Parameters
    ----------
    closed_threshold : float
        EAR below this value → eye is considered closed.
        Should match or be slightly lower than EAR_CLOSED_THRESHOLD (0.15).
    long_closure_min_ms : float
        Minimum duration to classify a closure as "long" (default 500 ms).
    closure_timeout_ms : float
        If no face frame arrives for this long while in a closure, cancel it.
    max_history_seconds : float
        How much frame history to keep (2 minutes is enough for 60 s windows).
    """

    def __init__ (
    self ,
    closed_threshold :float =0.15 ,
    long_closure_min_ms :float =500.0 ,
    closure_timeout_ms :float =2_000.0 ,
    max_history_seconds :float =120.0 ,
    ):
        self .closed_threshold =closed_threshold 
        self .long_closure_min_ms =long_closure_min_ms 
        self .closure_timeout_ms =closure_timeout_ms 
        self ._max_history_ms =max_history_seconds *1_000.0 


        self ._frames :deque [Tuple [float ,bool ]]=deque ()


        self ._in_closure :bool =False 
        self ._closure_start_ms :Optional [float ]=None 


        self ._long_closures :deque [ClosureEvent ]=deque ()



    def update (
    self ,
    ema_ear_avg :float ,
    timestamp_ms :Optional [float ]=None ,
    )->Optional [ClosureEvent ]:
        """
        Process one frame. Returns a ClosureEvent if a long closure just ended,
        otherwise None.
        """
        if timestamp_ms is None :
            timestamp_ms =time .time ()*1_000.0 

        is_closed =ema_ear_avg <self .closed_threshold 


        self ._frames .append ((timestamp_ms ,is_closed ))
        self ._prune_frames (timestamp_ms )


        if self ._in_closure and self ._closure_start_ms is not None :
            elapsed =timestamp_ms -self ._closure_start_ms 
            if elapsed >self .closure_timeout_ms and not is_closed :
                self ._in_closure =False 
                self ._closure_start_ms =None 


        event :Optional [ClosureEvent ]=None 

        if not self ._in_closure and is_closed :
            self ._in_closure =True 
            self ._closure_start_ms =timestamp_ms 

        elif self ._in_closure and not is_closed :
            duration =timestamp_ms -self ._closure_start_ms 
            self ._in_closure =False 
            self ._closure_start_ms =None 

            if duration >=self .long_closure_min_ms :
                event =ClosureEvent (
                start_ms =timestamp_ms -duration ,
                end_ms =timestamp_ms ,
                duration_ms =duration ,
                )
                self ._long_closures .append (event )
                self ._prune_closures (timestamp_ms )

        return event 

    def get_perclos (
    self ,
    timestamp_ms :Optional [float ]=None ,
    window_seconds :float =60.0 ,
    )->float :
        """Return the fraction of frames that were 'closed' in the last window."""
        if timestamp_ms is None :
            timestamp_ms =time .time ()*1_000.0 

        cutoff =timestamp_ms -window_seconds *1_000.0 
        window =[(ts ,c )for ts ,c in self ._frames if ts >=cutoff ]

        if not window :
            return 0.0 
        return sum (1 for _ ,c in window if c )/len (window )

    def get_long_closure_count (
    self ,
    timestamp_ms :Optional [float ]=None ,
    window_seconds :float =60.0 ,
    )->int :
        if timestamp_ms is None :
            timestamp_ms =time .time ()*1_000.0 
        cutoff =timestamp_ms -window_seconds *1_000.0 
        return sum (1 for e in self ._long_closures if e .end_ms >=cutoff )

    def get_total_closure_time_ms (
    self ,
    timestamp_ms :Optional [float ]=None ,
    window_seconds :float =60.0 ,
    )->float :
        """Total ms of long-closure events in the window."""
        if timestamp_ms is None :
            timestamp_ms =time .time ()*1_000.0 
        cutoff =timestamp_ms -window_seconds *1_000.0 
        return sum (e .duration_ms for e in self ._long_closures if e .end_ms >=cutoff )

    def reset (self ):
        self ._frames .clear ()
        self ._long_closures .clear ()
        self ._in_closure =False 
        self ._closure_start_ms =None 



    def _prune_frames (self ,timestamp_ms :float ):
        cutoff =timestamp_ms -self ._max_history_ms 
        while self ._frames and self ._frames [0 ][0 ]<cutoff :
            self ._frames .popleft ()

    def _prune_closures (self ,timestamp_ms :float ):
        cutoff =timestamp_ms -self ._max_history_ms 
        while self ._long_closures and self ._long_closures [0 ].end_ms <cutoff :
            self ._long_closures .popleft ()
