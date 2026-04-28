"""JSON serialization for CV-engine frame data."""

import json 
from typing import Optional 

from app .metrics .result import EyeMetrics
from app .state .distraction import DistractionResult
from app .state .cv_state import CVState
from app .emotions .result import EmotionResult


def serialize_frame (
timestamp_ms :float ,
cv_state :CVState ,
metrics :Optional [EyeMetrics ],
distraction :DistractionResult ,
emotion :Optional [EmotionResult ]=None ,
)->str :
    """Serialize one frame snapshot to a JSON string."""
    payload :dict ={
    "type":"frame",
    "timestamp_ms":round (timestamp_ms ,1 ),

    "eye_state":cv_state .eye_state .value ,
    "fatigue_level":cv_state .fatigue_level .value ,
    "attention":cv_state .attention .value ,

    "metrics":_metrics_dict (metrics ),
    "emotion":_emotion_dict (emotion ),

    "distraction":{
    "face_present":distraction .face_present ,
    "face_absent_ms":round (distraction .face_absent_ms ,1 ),
    "face_absent_alert":distraction .face_absent_alert ,
    "yaw":round (distraction .yaw ,2 ),
    "pitch":round (distraction .pitch ,2 ),
    "roll":round (distraction .roll ,2 ),
    "is_looking_away":distraction .is_looking_away ,
    "is_head_tilted":distraction .is_head_tilted ,
    "is_distracted":distraction .is_distracted ,
    "distraction_score":round (distraction .distraction_score ,3 ),
    "reason":distraction .reason .value ,
    },
    }
    return json .dumps (payload ,separators =(",",":"))


def serialize_event (event_type :str ,**kwargs )->str :
    """Serialize a control event (session_started, session_stopped, etc.)."""
    return json .dumps ({"type":event_type ,**kwargs },separators =(",",":"))


def _emotion_dict (e :Optional [EmotionResult ])->Optional [dict ]:
    if e is None :
        return None
    return {
    "emotion":e .emotion .value ,
    "confidence":round (e .confidence ,3 ),
    "scores":{k .value :round (v ,3 )for k ,v in e .scores .items ()},
    "mar":round (e .mar ,4 ),
    "smile":round (e .smile ,4 ),
    "brow_raise":round (e .brow_raise ,4 ),
    "brow_tilt":round (e .brow_tilt ,4 ),
    }


def _metrics_dict (m :Optional [EyeMetrics ])->Optional [dict ]:
    if m is None :
        return None 
    return {
    "ear_avg":round (m .ear_avg_smooth ,4 ),
    "ear_baseline":round (m .ear_baseline ,4 ),
    "ear_deviation":round (m .ear_deviation ,4 ),
    "blink_detected":m .blink_detected ,
    "blink_duration_ms":round (m .blink_duration_ms ,1 ),
    "blink_rate_30s":int (m .blink_rate_30s ),
    "blink_rate_60s":int (m .blink_rate_60s ),
    "avg_blink_duration_ms":round (m .avg_blink_duration_ms ,1 ),
    "long_blink_count":m .long_blink_count ,
    "perclos_30s":round (m .perclos_30s ,4 ),
    "perclos_60s":round (m .perclos_60s ,4 ),
    "fatigue_score":round (m .fatigue_score ,3 ),
    }
