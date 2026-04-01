"""
Rule-based fatigue scorer — Stage 4.

Three independent components, each mapped to [0, 1]:

  perclos_component       — fraction of 60 s spent with eyes closed
  long_closure_component  — count of closures > 500 ms in last 60 s
  blink_duration_component — how much avg blink duration exceeds a normal baseline

Weighted sum → fatigue_score ∈ [0, 1] → FatigueLevel.

Component weights and saturation thresholds are configurable so they can be
tuned from calibration data without changing the logic.
"""

from dataclasses import dataclass 
from typing import Optional 

from app .metrics .result import FatigueLevel 









PERCLOS_SATURATION =0.30 


NORMAL_BLINK_DURATION_MS =150.0 


LONG_CLOSURE_SATURATION =3 


@dataclass (frozen =True )
class FatigueComponents :
    """Individual [0, 1] contributions before weighting."""
    perclos :float 
    long_closure :float 
    blink_duration :float 


@dataclass (frozen =True )
class FatigueResult :
    score :float 
    level :FatigueLevel 
    components :FatigueComponents 


class FatigueScorer :
    """
    Rule-based fatigue scorer.

    Parameters
    ----------
    perclos_weight, long_closure_weight, blink_duration_weight : float
        Component weights; must sum to 1.0.
    perclos_sat : float
        PERCLOS value at which the perclos component = 1.0.
    long_closure_sat : int
        Long-closure count at which that component = 1.0.
    normal_blink_ms : float
        "Alert" baseline for average blink duration.
    blink_duration_sat_ratio : float
        Fractional increase in avg blink duration at which component = 1.0.
        Default 1.0 → a 100 % increase over baseline saturates the component.
    """

    def __init__ (
    self ,
    perclos_weight :float =0.50 ,
    long_closure_weight :float =0.30 ,
    blink_duration_weight :float =0.20 ,
    perclos_sat :float =PERCLOS_SATURATION ,
    long_closure_sat :int =LONG_CLOSURE_SATURATION ,
    normal_blink_ms :float =NORMAL_BLINK_DURATION_MS ,
    blink_duration_sat_ratio :float =1.0 ,
    ):
        total =perclos_weight +long_closure_weight +blink_duration_weight 
        if abs (total -1.0 )>1e-6 :
            raise ValueError (f"Weights must sum to 1.0, got {total :.4f}")

        self .perclos_weight =perclos_weight 
        self .long_closure_weight =long_closure_weight 
        self .blink_duration_weight =blink_duration_weight 
        self .perclos_sat =perclos_sat 
        self .long_closure_sat =long_closure_sat 
        self .normal_blink_ms =normal_blink_ms 
        self .blink_duration_sat_ratio =blink_duration_sat_ratio 

    def score (
    self ,
    perclos :float ,
    long_closure_count :int ,
    avg_blink_duration_ms :float ,
    )->FatigueResult :
        """
        Compute fatigue score from three inputs.

        Parameters
        ----------
        perclos : float
            PERCLOS value in [0, 1] (60-second window).
        long_closure_count : int
            Number of prolonged (> 500 ms) eye closures in the last 60 s.
        avg_blink_duration_ms : float
            Average blink duration in the last 60 s. Pass 0.0 if no blinks.
        """
        p =_clamp (perclos /self .perclos_sat )

        l =_clamp (long_closure_count /self .long_closure_sat )

        if avg_blink_duration_ms >0.0 :
            excess =(avg_blink_duration_ms -self .normal_blink_ms )/(
            self .normal_blink_ms *self .blink_duration_sat_ratio 
            )
            d =_clamp (excess )
        else :
            d =0.0 

        raw =(
        self .perclos_weight *p 
        +self .long_closure_weight *l 
        +self .blink_duration_weight *d 
        )
        final =_clamp (raw )

        return FatigueResult (
        score =final ,
        level =FatigueLevel .from_score (final ),
        components =FatigueComponents (perclos =p ,long_closure =l ,blink_duration =d ),
        )




def _clamp (value :float ,lo :float =0.0 ,hi :float =1.0 )->float :
    return max (lo ,min (hi ,value ))
