"""
Background QThread that owns EngineSession and emits UI-safe frame summaries.

Heavy ``app.*`` / MediaPipe imports are deferred until ``run()`` so the GUI can
start without loading the CV stack on the main thread.
"""

from __future__ import annotations 

import time 
from typing import Any ,Dict ,Union 

import numpy as np 
from PySide6 .QtCore import QMutex ,QMutexLocker ,QThread ,Signal 
from PySide6 .QtGui import QImage 

from desktop_app .alert_policy import AlertPolicy 


def _bgr_numpy_to_qimage_copy (frame :np .ndarray )->QImage :
    """BGR uint8 → RGB QImage (deep copy, safe across threads)."""
    import cv2 

    rgb =cv2 .cvtColor (frame ,cv2 .COLOR_BGR2RGB )
    rgb =np .ascontiguousarray (rgb )
    h ,w ,_ =rgb .shape 
    bpl =3 *w 
    return QImage (rgb .tobytes (),w ,h ,bpl ,QImage .Format .Format_RGB888 ).copy ()


class EngineThread (QThread ):
    """Owns EngineSession entirely on this thread; headless loop only."""

    frame_summary =Signal (dict )
    preview_frame =Signal (QImage )
    alert_raised =Signal (dict )
    failed =Signal (str )
    finished_clean =Signal ()

    def __init__ (
    self ,
    camera_source :Union [int ,str ]=0 ,
    parent =None ,
    *,
    preview_interval_ms :float =66.0 ,
    ):
        super ().__init__ (parent )
        self ._camera_source =camera_source 
        self ._preview_interval_ms =preview_interval_ms 
        self ._stop_requested =False 
        self ._pause_mutex =QMutex ()
        self ._pause_requested =False 

    def request_stop (self )->None :
        self ._stop_requested =True 

    def set_paused (self ,paused :bool )->None :
        with QMutexLocker (self ._pause_mutex ):
            self ._pause_requested =paused 

    def _is_paused (self )->bool :
        with QMutexLocker (self ._pause_mutex ):
            return self ._pause_requested 

    def run (self )->None :
        from desktop_app .log_silence import apply_log_silence ,mediapipe_stderr_filter 

        apply_log_silence ()

        with mediapipe_stderr_filter ():
            from app .engine .session import EngineSession 

            def ui_payload_from_frame (
            result :Any ,
            blink_total :int ,
            closure_streak_ms :float ,
            )->Dict [str ,Any ]:
                d =result .detection
                m =result .metrics
                r =result .distraction
                em =result .emotion
                payload :Dict [str ,Any ]={
                "frame_index":result .frame_index ,
                "detection_status":d .status .value ,
                "confidence":float (d .confidence ),
                "blink_this_frame":bool (m .blink_detected )if m else False ,
                "blink_total":blink_total ,
                "blink_duration_ms":float (m .blink_duration_ms )if m else 0.0 ,
                "perclos_30s":float (m .perclos_30s )if m else None ,
                "fatigue_level":m .fatigue_level .name if m else None ,
                "fatigue_score":float (m .fatigue_score )if m else None ,
                "distraction_reason":r .reason .value ,
                "distraction_score":float (r .distraction_score ),
                "is_distracted":bool (r .is_distracted ),
                "face_present":bool (r .face_present ),
                "face_absent_ms":float (r .face_absent_ms ),
                "closure_streak_ms":closure_streak_ms ,
                "state_left":m .state_left .name if m else None ,
                "state_right":m .state_right .name if m else None ,
                "emotion":em .emotion .value if em is not None else None ,
                "emotion_confidence":float (em .confidence )if em is not None else 0.0 ,
                "emotion_scores":(
                {e .value :float (v )for e ,v in em .scores .items ()}
                if em is not None else {}
                ),
                }
                return payload

            self ._stop_requested =False 
            session =None 
            blink_total =0 
            closure_streak_ms =0.0 
            last_preview_mono =0.0 
            policy =AlertPolicy (cooldown_s =45.0 )

            try :
                session =EngineSession (self ._camera_source ,auto_calibration =False )
            except Exception as e :
                self .failed .emit (str (e ))
                self .finished_clean .emit ()
                return 

            try :
                while not self ._stop_requested and session .capture .is_opened ():
                    if self ._is_paused ():
                        time .sleep (0.05 )
                        continue 

                    result =session .read_and_process (draw_overlays =False )
                    if result is None :
                        break 

                    if result .metrics is not None and result .metrics .blink_detected :
                        blink_total +=1 

                    m =result .metrics 
                    if m is not None :
                        if m .state_left .name =="CLOSED"and m .state_right .name =="CLOSED":
                            closure_streak_ms +=33.0 
                        else :
                            closure_streak_ms =0.0 
                    else :
                        closure_streak_ms =0.0 

                    payload =ui_payload_from_frame (result ,blink_total ,closure_streak_ms )
                    self .frame_summary .emit (payload )

                    alert =policy .evaluate (payload )
                    if alert is not None :
                        self .alert_raised .emit (alert .as_dict ())

                    now =time .monotonic ()
                    if (now -last_preview_mono )*1000.0 >=self ._preview_interval_ms :
                        last_preview_mono =now 
                        try :
                            qimg =_bgr_numpy_to_qimage_copy (result .frame )
                            self .preview_frame .emit (qimg )
                        except Exception :
                            pass 
            except Exception as e :
                self .failed .emit (str (e ))
            finally :
                if session is not None :
                    session .release ()
                self .finished_clean .emit ()
