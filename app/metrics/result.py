"""Output types for eye and fatigue metrics (Stages 2–4)."""

from dataclasses import dataclass 
from enum import Enum 

from app .metrics .ear import EyeState 




class FatigueLevel (Enum ):
    ALERT ="alert"
    MILD ="mild"
    MODERATE ="moderate"
    SEVERE ="severe"

    @classmethod 
    def from_score (cls ,score :float )->"FatigueLevel":
        if score >=0.75 :
            return cls .SEVERE 
        if score >=0.50 :
            return cls .MODERATE 
        if score >=0.25 :
            return cls .MILD 
        return cls .ALERT 

    @property 
    def colour_bgr (self )->tuple :
        """OpenCV BGR colour for overlays."""
        return {
        FatigueLevel .ALERT :(0 ,220 ,0 ),
        FatigueLevel .MODERATE :(0 ,165 ,255 ),
        FatigueLevel .MILD :(0 ,220 ,220 ),
        FatigueLevel .SEVERE :(0 ,0 ,220 ),
        }[self ]




@dataclass 
class EyeMetrics :

    ear_left :float 
    ear_right :float 
    ear_avg :float 


    ear_left_smooth :float 
    ear_right_smooth :float 
    ear_avg_smooth :float 


    ear_baseline :float 
    ear_deviation :float 


    state_left :EyeState 
    state_right :EyeState 


    blink_detected :bool 
    blink_duration_ms :float 


    blink_rate_30s :float 
    blink_rate_60s :float 
    avg_blink_duration_ms :float 
    long_blink_count :int 


    perclos_30s :float 
    perclos_60s :float 


    long_closure_count_60s :int 
    total_closure_time_60s_ms :float 


    fatigue_score :float 
    fatigue_level :FatigueLevel 
