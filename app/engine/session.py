"""
Reusable CV engine session: one owner for capture + detector + metrics + distraction.

Preview (OpenCV window) and headless loops share the same frame-processing path.
"""

from __future__ import annotations 

import time 
from dataclasses import dataclass 
from typing import Callable ,Optional ,Tuple ,Union 

import cv2 
import numpy as np 

from app .camera .capture import CameraCapture
from app .emotions .detector import EmotionDetector
from app .emotions .result import EmotionResult
from app .landmarks .detector import LandmarkDetector
from app .landmarks .result import DetectionResult ,DetectionStatus ,HeadPose
from app .metrics .calculator import MetricsCalculator
from app .metrics .result import EyeMetrics
from app .state .calibration import CalibrationSession ,CalibrationState
from app .state .distraction import DistractionResult ,DistractionTracker


@dataclass 
class DetectionSummary :
    """Per-frame detection snapshot without MediaPipe-specific objects."""

    status :DetectionStatus 
    confidence :float 
    face_bbox :Optional [Tuple [int ,int ,int ,int ]]=None 
    head_pose :Optional [HeadPose ]=None 


@dataclass 
class FrameProcessingResult :
    """Structured output for one processed frame (UI / worker / logging)."""

    frame :np .ndarray 
    detection :DetectionSummary 
    metrics :Optional [EyeMetrics ]
    distraction :DistractionResult
    timestamp_ms :float
    frame_index :int
    calibration_state :Optional [CalibrationState ]=None
    calibration_progress :float =0.0
    emotion :Optional [EmotionResult ]=None


def _summarize_detection (result :DetectionResult )->DetectionSummary :
    return DetectionSummary (
    status =result .status ,
    confidence =result .confidence ,
    face_bbox =result .face_bbox ,
    head_pose =result .head_pose ,
    )


class EngineSession :
    """
    Owns the full processing pipeline for one capture source.

    Safe to drive from a dedicated worker thread later: keep one session
    instance per thread and do not share it across threads.
    """

    def __init__ (
    self ,
    source :Union [int ,str ],
    *,
    auto_calibration :bool =True ,
    calibration_duration_s :float =30.0 ,
    calibration_min_valid_s :float =12.0 ,
    ):
        self ._capture =CameraCapture (source =source )
        self ._detector =LandmarkDetector ()
        self ._calculator =MetricsCalculator ()
        self ._distraction =DistractionTracker ()
        self ._emotions =EmotionDetector ()
        self ._frame_index =0
        self ._calibration :Optional [CalibrationSession ]=None 
        if auto_calibration :
            self ._calibration =CalibrationSession (
            duration_seconds =calibration_duration_s ,
            min_valid_seconds =calibration_min_valid_s ,
            )

    @property 
    def capture (self )->CameraCapture :
        return self ._capture 

    @property 
    def detector (self )->LandmarkDetector :
        return self ._detector 

    @property 
    def calculator (self )->MetricsCalculator :
        return self ._calculator 

    @property 
    def distraction (self )->DistractionTracker :
        return self ._distraction 

    def read_and_process (
    self ,
    *,
    draw_overlays :bool =False ,
    timestamp_ms :Optional [float ]=None ,
    )->Optional [FrameProcessingResult ]:
        """
        Read one frame from capture and run detection → metrics → distraction.

        Mutates ``frame`` in place when ``draw_overlays`` is True (same as legacy CLI).
        Returns None when capture is closed or read fails.
        """
        if not self ._capture .is_opened ():
            return None 

        frame =self ._capture .read ()
        if frame is None :
            return None 

        ts =timestamp_ms if timestamp_ms is not None else time .time ()*1_000.0 
        idx =self ._frame_index 
        self ._frame_index +=1 

        detection =self ._detector .detect (frame )
        metrics =self ._calculator .update (detection ,timestamp_ms =ts )
        emotion =self ._emotions .detect (detection )

        cal_state :Optional [CalibrationState ]=None 
        cal_progress =0.0 
        suppress_distr =False 

        if self ._calibration is not None :
            c =self ._calibration 
            if c .state ==CalibrationState .IDLE :
                c .start (ts )
            cal_progress =c .progress_at (ts )
            cal_state =c .state 
            if c .state ==CalibrationState .RUNNING :
                suppress_distr =True 
                nst =c .update (detection ,metrics ,timestamp_ms =ts )
                cal_state =nst 
                cal_progress =c .progress_at (ts )
                if nst ==CalibrationState .DONE :
                    prof =c .finish ()
                    if prof :
                        self ._calculator .apply_calibration (prof )
                        self ._distraction .apply_calibration (prof )
                    self ._calibration =None 
                elif nst ==CalibrationState .FAILED :
                    self ._calibration =None 

        distr =self ._distraction .update (
        detection ,
        timestamp_ms =ts ,
        metrics =metrics ,
        calibration_suppress =suppress_distr ,
        )

        if draw_overlays :
            self ._detector .draw (frame ,detection )
            self ._calculator .draw (frame ,metrics )
            self ._distraction .draw (frame ,distr )
            self ._emotions .draw (frame ,emotion )
            if self ._calibration is not None :
                self ._calibration .draw (frame ,timestamp_ms =ts )

        return FrameProcessingResult (
        frame =frame ,
        detection =_summarize_detection (detection ),
        metrics =metrics ,
        distraction =distr ,
        timestamp_ms =ts ,
        frame_index =idx ,
        calibration_state =cal_state ,
        calibration_progress =cal_progress ,
        emotion =emotion ,
        )

    def run_loop (
    self ,
    *,
    preview :bool =False ,
    draw_overlays :Optional [bool ]=None ,
    on_frame :Optional [Callable [[FrameProcessingResult ],None ]]=None ,
    stop_check :Optional [Callable [[],bool ]]=None ,
    )->None :
        """
        Process frames until capture ends, user presses 'q' (preview only), or stop_check.

        When ``preview`` is True, shows an OpenCV window and handles quit via 'q'.
        When ``draw_overlays`` is None, it defaults to ``preview`` (legacy CLI behavior).
        """
        if draw_overlays is None :
            draw_overlays =preview 

        window_name ="CV Engine"
        while self ._capture .is_opened ():
            if stop_check is not None and stop_check ():
                break 

            result =self .read_and_process (draw_overlays =draw_overlays )
            if result is None :
                break 

            if on_frame is not None :
                on_frame (result )

            if preview :
                cv2 .imshow (window_name ,result .frame )
                if cv2 .waitKey (1 )&0xFF ==ord ("q"):
                    break 

    def release (self )->None :
        self ._capture .release ()
        self ._detector .release ()
        cv2 .destroyAllWindows ()
