"""
EAR temporal smoothing — Stage 3.

Two complementary mechanisms:
  1. EMA (Exponential Moving Average): suppresses single-frame noise.
     alpha=0.15 → a lone noise spike shifts the EMA by only ~15%, not enough
     to cross the blink threshold (0.20) from a normal open-eye baseline (0.30+).
  2. Rolling window baseline: tracks the person's personal open-eye EAR
     over the last N seconds, used downstream to detect persistent closure.
"""

import time 
from collections import deque 
from dataclasses import dataclass 
from typing import Optional ,Tuple 


@dataclass 
class SmoothEAR :
    """Output of EarSmoother.update()."""

    raw_left :float 
    raw_right :float 
    raw_avg :float 


    ema_left :float 
    ema_right :float 
    ema_avg :float 


    baseline :float 
    deviation :float 

    timestamp_ms :float 


class EarSmoother :
    """
    Applies EMA and maintains a rolling baseline for raw EAR values.

    Parameters
    ----------
    alpha : float
        EMA coefficient in (0, 1].  Lower = smoother, more lag.
        Default 0.15 is appropriate for 30 fps.
    window_seconds : float
        Duration of the rolling window for the baseline computation.
        Should cover several blink cycles (~4 s).
    """

    def __init__ (self ,alpha :float =0.15 ,window_seconds :float =4.0 ):
        if not 0.0 <alpha <=1.0 :
            raise ValueError (f"alpha must be in (0, 1], got {alpha }")
        self .alpha =alpha 
        self .window_ms =window_seconds *1_000.0 

        self ._ema_left :Optional [float ]=None 
        self ._ema_right :Optional [float ]=None 

        self ._window :deque [Tuple [float ,float ]]=deque ()



    def update (
    self ,
    ear_left :float ,
    ear_right :float ,
    timestamp_ms :Optional [float ]=None ,
    )->SmoothEAR :
        if timestamp_ms is None :
            timestamp_ms =time .time ()*1_000.0 


        if self ._ema_left is None :

            self ._ema_left =ear_left 
            self ._ema_right =ear_right 
        else :
            self ._ema_left =self .alpha *ear_left +(1.0 -self .alpha )*self ._ema_left 
            self ._ema_right =self .alpha *ear_right +(1.0 -self .alpha )*self ._ema_right 

        ema_avg =(self ._ema_left +self ._ema_right )/2.0 


        self ._window .append ((timestamp_ms ,ema_avg ))
        self ._prune (timestamp_ms )

        baseline =sum (v for _ ,v in self ._window )/len (self ._window )
        deviation =baseline -ema_avg 

        return SmoothEAR (
        raw_left =ear_left ,
        raw_right =ear_right ,
        raw_avg =(ear_left +ear_right )/2.0 ,
        ema_left =self ._ema_left ,
        ema_right =self ._ema_right ,
        ema_avg =ema_avg ,
        baseline =baseline ,
        deviation =deviation ,
        timestamp_ms =timestamp_ms ,
        )

    def reset (self ):
        self ._ema_left =None 
        self ._ema_right =None 
        self ._window .clear ()

    @property 
    def is_warmed_up (self )->bool :
        """True after at least one frame has been processed."""
        return self ._ema_left is not None 



    def _prune (self ,timestamp_ms :float ):
        cutoff =timestamp_ms -self .window_ms 
        while self ._window and self ._window [0 ][0 ]<cutoff :
            self ._window .popleft ()
