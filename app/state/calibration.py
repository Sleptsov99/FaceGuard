"""
User calibration — Stage 6.

A baseline session (desktop: **30 s** by default) while the user looks at the
screen normally.  The session collects:
  - open-eye EAR distribution   → personalised closed / blink EAR thresholds
  - blink duration distribution → personalised long-blink threshold
  - head pose distribution      → personalised looking-away thresholds
  - iris position along each eye (478 mesh) → personalised gaze deviation threshold

Thresholds are derived as  mean ± k·σ  and clamped to physiologically
reasonable ranges so extreme faces or head positions can't produce
nonsensical thresholds.

Usage
-----
session = CalibrationSession(duration_seconds=45)

# In the frame loop:
state = session.update(detection, metrics, timestamp_ms=ts)
if state == CalibrationState.DONE:
    profile = session.finish()
    calculator.apply_calibration(profile)
    distraction.apply_calibration(profile)
"""

import math 
import time 
from dataclasses import dataclass 
from enum import Enum 
from typing import List ,Optional 

import cv2 
import numpy as np 

from app .landmarks .result import DetectionResult ,DetectionStatus 
from app .metrics .ear import EyeState 
from app .metrics .result import EyeMetrics 




class CalibrationState (Enum ):
    IDLE ="idle"
    RUNNING ="running"
    DONE ="done"
    FAILED ="failed"


@dataclass (frozen =True )
class CalibrationProfile :
    """Personal thresholds derived from a calibration session."""


    ear_mean :float 
    ear_std :float 
    ear_closed_threshold :float 
    ear_blink_threshold :float 


    blink_duration_mean_ms :float 
    blink_duration_std_ms :float 
    long_blink_threshold_ms :float 


    yaw_mean :float 
    pitch_mean :float 
    roll_mean :float 
    pose_std :float 
    yaw_threshold :float 
    pitch_threshold :float 
    roll_threshold :float 


    gaze_t_mean_right :float 
    gaze_t_mean_left :float 
    gaze_threshold :float 
    gaze_enabled :bool 


    sample_count :int 
    duration_ms :float 

    @property 
    def is_valid (self )->bool :
        return self .sample_count >0 




class CalibrationSession :
    """
    Collects samples and computes a CalibrationProfile.

    Parameters
    ----------
    duration_seconds : float
        Target session length (shows progress bar up to this).
    min_valid_seconds : float
        Minimum of valid OK frames required; session FAILs otherwise.
    k_ear : float
        Sigma multiplier for EAR thresholds (default 2.0).
    k_pose : float
        Sigma multiplier for head-pose thresholds (default 2.0).
    min_pose_margin : float
        Minimum degrees added beyond mean for pose thresholds.
    k_gaze : float
        Множитель σ для порога отклонения взгляда (меньше → чувствительнее).
    min_gaze_margin : float
        Минимальный порог в t-пространстве [0,1] поверх k·σ.
    """

    def __init__ (
    self ,
    duration_seconds :float =45.0 ,
    min_valid_seconds :float =20.0 ,
    k_ear :float =2.0 ,
    k_pose :float =2.0 ,
    min_pose_margin :float =15.0 ,
    k_gaze :float =2.05 ,
    min_gaze_margin :float =0.055 ,
    ):
        self .duration_ms =duration_seconds *1_000.0 
        self .min_valid_ms =min_valid_seconds *1_000.0 
        self .k_ear =k_ear 
        self .k_pose =k_pose 
        self .min_pose_margin =min_pose_margin 
        self .k_gaze =k_gaze 
        self .min_gaze_margin =min_gaze_margin 

        self ._state =CalibrationState .IDLE 
        self ._start_ms :Optional [float ]=None 
        self ._profile :Optional [CalibrationProfile ]=None 


        self ._ear_samples :List [float ]=[]
        self ._blink_durations :List [float ]=[]
        self ._yaw_samples :List [float ]=[]
        self ._pitch_samples :List [float ]=[]
        self ._roll_samples :List [float ]=[]
        self ._gaze_right_t :List [float ]=[]
        self ._gaze_left_t :List [float ]=[]



    def start (self ,timestamp_ms :Optional [float ]=None ):
        """Begin (or restart) the calibration session."""
        if timestamp_ms is None :
            timestamp_ms =time .time ()*1_000.0 
        self ._start_ms =timestamp_ms 
        self ._state =CalibrationState .RUNNING 
        self ._profile =None 
        self ._ear_samples .clear ()
        self ._blink_durations .clear ()
        self ._yaw_samples .clear ()
        self ._pitch_samples .clear ()
        self ._roll_samples .clear ()
        self ._gaze_right_t .clear ()
        self ._gaze_left_t .clear ()

    def update (
    self ,
    detection :DetectionResult ,
    metrics :Optional [EyeMetrics ],
    timestamp_ms :Optional [float ]=None ,
    )->CalibrationState :
        """
        Feed one frame.  Returns the current CalibrationState.
        Only call after start().
        """
        if self ._state !=CalibrationState .RUNNING :
            return self ._state 
        if timestamp_ms is None :
            timestamp_ms =time .time ()*1_000.0 

        elapsed =timestamp_ms -self ._start_ms 


        if detection .status ==DetectionStatus .OK and metrics is not None :

            if (metrics .state_left ==EyeState .OPEN 
            and metrics .state_right ==EyeState .OPEN ):
                self ._ear_samples .append (metrics .ear_avg_smooth )


            if metrics .blink_detected and metrics .blink_duration_ms >0 :
                self ._blink_durations .append (metrics .blink_duration_ms )


            if detection .head_pose is not None :
                self ._yaw_samples .append (detection .head_pose .yaw )
                self ._pitch_samples .append (detection .head_pose .pitch )
                self ._roll_samples .append (detection .head_pose .roll )


            hp =detection .head_pose 
            if (
            metrics .state_left ==EyeState .OPEN 
            and metrics .state_right ==EyeState .OPEN 
            and detection .gaze_iris_t is not None 
            and hp is not None 
            and hp .is_frontal (22.0 ,22.0 )
            ):
                tr ,tl =detection .gaze_iris_t 
                self ._gaze_right_t .append (tr )
                self ._gaze_left_t .append (tl )


        if elapsed >=self .duration_ms :
            valid_ms =len (self ._ear_samples )*33.0 
            if valid_ms >=self .min_valid_ms :
                self ._state =CalibrationState .DONE 
            else :
                self ._state =CalibrationState .FAILED 

        return self ._state 

    def finish (self )->Optional [CalibrationProfile ]:
        """
        Compute and return the CalibrationProfile.
        Returns None if the session has not completed successfully.
        Caches the result so repeated calls are free.
        """
        if self ._state !=CalibrationState .DONE :
            return None 
        if self ._profile is not None :
            return self ._profile 
        self ._profile =self ._compute ()
        return self ._profile 

    @property 
    def state (self )->CalibrationState :
        return self ._state 

    @property 
    def progress (self )->float :
        """[0, 1] fraction of session elapsed."""
        if self ._state ==CalibrationState .IDLE or self ._start_ms is None :
            return 0.0 
        if self ._state in (CalibrationState .DONE ,CalibrationState .FAILED ):
            return 1.0 
        elapsed =time .time ()*1_000.0 -self ._start_ms 
        return min (elapsed /self .duration_ms ,1.0 )

    def progress_at (self ,timestamp_ms :float )->float :
        """Progress at a specific timestamp (for deterministic tests)."""
        if self ._state ==CalibrationState .IDLE or self ._start_ms is None :
            return 0.0 
        elapsed =timestamp_ms -self ._start_ms 
        return min (elapsed /self .duration_ms ,1.0 )

    def draw (self ,frame :np .ndarray ,timestamp_ms :Optional [float ]=None )->np .ndarray :
        """Draw calibration overlay (progress bar + instructions)."""
        h ,w =frame .shape [:2 ]

        if self ._state ==CalibrationState .IDLE :
            return frame 

        prog =self .progress_at (timestamp_ms )if timestamp_ms else self .progress 

        if self ._state ==CalibrationState .RUNNING :
            colour =(0 ,200 ,200 )
            label =f"Calibrating...  {int (prog *100 )}%"

            big ="CALIBRATING"
            (bw ,bh ),_ =cv2 .getTextSize (big ,cv2 .FONT_HERSHEY_SIMPLEX ,1.0 ,2 )
            bx =(w -bw )//2 
            by =h //2 
            cv2 .rectangle (frame ,(bx -12 ,by -bh -10 ),(bx +bw +12 ,by +10 ),
            (30 ,30 ,30 ),-1 )
            cv2 .putText (frame ,big ,(bx ,by ),cv2 .FONT_HERSHEY_SIMPLEX ,
            1.0 ,colour ,2 ,cv2 .LINE_AA )
            hint2 ="Look at the screen normally"
            (hw2 ,hh2 ),_ =cv2 .getTextSize (hint2 ,cv2 .FONT_HERSHEY_SIMPLEX ,0.55 ,1 )
            hx2 =(w -hw2 )//2 
            cv2 .putText (frame ,hint2 ,(hx2 ,by +hh2 +18 ),
            cv2 .FONT_HERSHEY_SIMPLEX ,0.55 ,colour ,1 ,cv2 .LINE_AA )
        elif self ._state ==CalibrationState .DONE :
            colour =(0 ,220 ,0 )
            label ="Calibration done  — personal thresholds applied"
        else :
            colour =(0 ,0 ,220 )
            label ="Calibration failed  — not enough valid frames"


        bar_y =h -8 
        cv2 .rectangle (frame ,(0 ,bar_y ),(w ,h ),(40 ,40 ,40 ),-1 )
        cv2 .rectangle (frame ,(0 ,bar_y ),(int (w *prog ),h ),colour ,-1 )


        cv2 .rectangle (frame ,(0 ,h -30 ),(w ,h -9 ),(30 ,30 ,30 ),-1 )
        cv2 .putText (frame ,label ,
        (10 ,h -14 ),cv2 .FONT_HERSHEY_SIMPLEX ,
        0.52 ,colour ,1 ,cv2 .LINE_AA )
        return frame 



    def _compute (self )->CalibrationProfile :

        ear_arr =np .array (self ._ear_samples ,dtype =np .float64 )
        ear_mean =float (np .mean (ear_arr ))
        ear_std =float (np .std (ear_arr ))

        ear_closed =_clamp (
        ear_mean -self .k_ear *ear_std ,
        lo =0.08 ,hi =0.20 ,
        )
        ear_blink =_clamp (
        ear_mean -1.5 *ear_std ,
        lo =0.10 ,hi =0.25 ,
        )

        ear_blink =max (ear_blink ,ear_closed +0.02 )


        if self ._blink_durations :
            dur_arr =np .array (self ._blink_durations ,dtype =np .float64 )
            dur_mean =float (np .mean (dur_arr ))
            dur_std =float (np .std (dur_arr ))
        else :
            dur_mean =150.0 
            dur_std =50.0 

        long_blink_ms =_clamp (
        dur_mean +2.0 *dur_std ,
        lo =300.0 ,hi =800.0 ,
        )


        yaw_arr =np .array (self ._yaw_samples ,dtype =np .float64 )
        pitch_arr =np .array (self ._pitch_samples ,dtype =np .float64 )
        roll_arr =np .array (self ._roll_samples ,dtype =np .float64 )

        yaw_mean =float (np .mean (yaw_arr ))
        pitch_mean =float (np .mean (pitch_arr ))
        roll_mean =float (np .mean (roll_arr ))

        yaw_std =float (np .std (yaw_arr ))
        pitch_std =float (np .std (pitch_arr ))
        roll_std =float (np .std (roll_arr ))
        pose_std =float (np .mean ([yaw_std ,pitch_std ,roll_std ]))


        yaw_thresh =_clamp (
        self .k_pose *yaw_std +self .min_pose_margin ,
        lo =15.0 ,hi =50.0 ,
        )
        pitch_thresh =_clamp (
        self .k_pose *pitch_std +self .min_pose_margin ,
        lo =15.0 ,hi =50.0 ,
        )
        roll_thresh =_clamp (
        self .k_pose *roll_std +self .min_pose_margin ,
        lo =10.0 ,hi =35.0 ,
        )

        gaze_mr =gaze_ml =0.5 
        gaze_thresh =1.0 
        gaze_on =False 
        n_gaze =min (len (self ._gaze_right_t ),len (self ._gaze_left_t ))
        if n_gaze >=35 :
            gr =np .array (self ._gaze_right_t [:n_gaze ],dtype =np .float64 )
            gl =np .array (self ._gaze_left_t [:n_gaze ],dtype =np .float64 )
            gaze_mr =float (np .mean (gr ))
            gaze_ml =float (np .mean (gl ))
            std_r =float (np .std (gr ))
            std_l =float (np .std (gl ))
            pooled =max (std_r ,std_l ,0.018 )
            gaze_thresh =_clamp (
            self .k_gaze *pooled +self .min_gaze_margin ,
            lo =0.09 ,
            hi =0.32 ,
            )
            gaze_on =True 

        return CalibrationProfile (
        ear_mean =ear_mean ,
        ear_std =ear_std ,
        ear_closed_threshold =ear_closed ,
        ear_blink_threshold =ear_blink ,
        blink_duration_mean_ms =dur_mean ,
        blink_duration_std_ms =dur_std ,
        long_blink_threshold_ms =long_blink_ms ,
        yaw_mean =yaw_mean ,
        pitch_mean =pitch_mean ,
        roll_mean =roll_mean ,
        pose_std =pose_std ,
        yaw_threshold =yaw_thresh ,
        pitch_threshold =pitch_thresh ,
        roll_threshold =roll_thresh ,
        gaze_t_mean_right =gaze_mr ,
        gaze_t_mean_left =gaze_ml ,
        gaze_threshold =gaze_thresh ,
        gaze_enabled =gaze_on ,
        sample_count =len (self ._ear_samples ),
        duration_ms =self .duration_ms ,
        )




def _clamp (value :float ,lo :float ,hi :float )->float :
    return max (lo ,min (hi ,value ))
