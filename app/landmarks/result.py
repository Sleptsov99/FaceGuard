"""Detection result types for Stage 1."""

from dataclasses import dataclass ,field 
from enum import Enum 
from typing import Optional ,Any ,Tuple 

import numpy as np 


class DetectionStatus (Enum ):
    OK ="ok"
    NO_FACE ="no_face"
    FACE_TOO_SMALL ="face_too_small"
    BAD_ANGLE ="bad_angle"
    MULTIPLE_FACES ="multiple_faces"

    def is_usable (self )->bool :
        """True when landmarks were extracted (even if quality is degraded)."""
        return self in (DetectionStatus .OK ,DetectionStatus .BAD_ANGLE ,DetectionStatus .MULTIPLE_FACES )


@dataclass 
class EyeLandmarks :
    """Eye landmark points in pixel coordinates."""
    contour :np .ndarray 
    ear_points :np .ndarray 


@dataclass 
class HeadPose :
    """Head orientation angles in degrees (right-hand convention)."""
    yaw :float 
    pitch :float 
    roll :float 

    def is_frontal (self ,max_yaw :float =30.0 ,max_pitch :float =30.0 )->bool :
        return abs (self .yaw )<=max_yaw and abs (self .pitch )<=max_pitch 


@dataclass 
class DetectionResult :
    status :DetectionStatus 
    confidence :float =0.0 
    face_bbox :Optional [Tuple [int ,int ,int ,int ]]=None 
    right_eye :Optional [EyeLandmarks ]=None 
    left_eye :Optional [EyeLandmarks ]=None 
    head_pose :Optional [HeadPose ]=None 
    raw_landmarks :Optional [Any ]=None 

    gaze_iris_t :Optional [Tuple [float ,float ]]=None 

    @property 
    def face_found (self )->bool :
        return self .status !=DetectionStatus .NO_FACE 

    @property 
    def is_valid (self )->bool :
        return self .status ==DetectionStatus .OK 
