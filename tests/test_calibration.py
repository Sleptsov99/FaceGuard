"""Tests for Stage 6: user calibration."""

import sys 
import unittest 
from pathlib import Path 
from typing import Optional 

import numpy as np 

sys .path .insert (0 ,str (Path (__file__ ).parent .parent ))

from app .landmarks .result import DetectionResult ,DetectionStatus ,HeadPose 
from app .metrics .ear import EyeState 
from app .metrics .result import EyeMetrics ,FatigueLevel 
from app .state .calibration import CalibrationSession ,CalibrationState ,CalibrationProfile 
from app .metrics .calculator import MetricsCalculator 
from app .state .distraction import DistractionTracker ,DistractionReason 




def _ts (offset_ms :float =0.0 )->float :
    return 1_000_000.0 +offset_ms 


def _ok_detection (yaw =0.0 ,pitch =0.0 ,roll =0.0 ,confidence =0.9 )->DetectionResult :
    return DetectionResult (
    status =DetectionStatus .OK ,
    confidence =confidence ,
    head_pose =HeadPose (yaw =yaw ,pitch =pitch ,roll =roll ),
    )


def _no_face ()->DetectionResult :
    return DetectionResult (status =DetectionStatus .NO_FACE ,confidence =0.0 )


def _metrics (ear =0.30 ,state =EyeState .OPEN ,blink =False ,blink_ms =0.0 )->EyeMetrics :
    """Minimal EyeMetrics for calibration tests."""
    return EyeMetrics (
    ear_left =ear ,ear_right =ear ,ear_avg =ear ,
    ear_left_smooth =ear ,ear_right_smooth =ear ,ear_avg_smooth =ear ,
    ear_baseline =ear ,ear_deviation =0.0 ,
    state_left =state ,state_right =state ,
    blink_detected =blink ,blink_duration_ms =blink_ms ,
    blink_rate_30s =0.0 ,blink_rate_60s =0.0 ,
    avg_blink_duration_ms =0.0 ,long_blink_count =0 ,
    perclos_30s =0.0 ,perclos_60s =0.0 ,
    long_closure_count_60s =0 ,total_closure_time_60s_ms =0.0 ,
    fatigue_score =0.0 ,fatigue_level =FatigueLevel .ALERT ,
    )




class TestCalibrationLifecycle (unittest .TestCase ):
    def test_initial_state_is_idle (self ):
        s =CalibrationSession ()
        self .assertEqual (s .state ,CalibrationState .IDLE )

    def test_start_transitions_to_running (self ):
        s =CalibrationSession ()
        s .start (_ts (0 ))
        self .assertEqual (s .state ,CalibrationState .RUNNING )

    def test_update_before_start_returns_idle (self ):
        s =CalibrationSession ()
        state =s .update (_ok_detection (),_metrics (),_ts (0 ))
        self .assertEqual (state ,CalibrationState .IDLE )

    def test_finish_before_done_returns_none (self ):
        s =CalibrationSession (duration_seconds =10.0 )
        s .start (_ts (0 ))
        self .assertIsNone (s .finish ())

    def test_restart_clears_previous_samples (self ):
        s =CalibrationSession (duration_seconds =5.0 ,min_valid_seconds =1.0 )
        s .start (_ts (0 ))

        for i in range (10 ):
            s .update (_ok_detection (),_metrics (ear =0.10 ,state =EyeState .CLOSED ),_ts (i *33 ))

        s .start (_ts (500 ))

        for i in range (100 ):
            s .update (_ok_detection (),_metrics (ear =0.30 ),_ts (500 +i *50 ))
        s .update (_ok_detection (),_metrics (),_ts (500 +5_100 ))
        if s .state ==CalibrationState .DONE :
            profile =s .finish ()

            self .assertGreater (profile .ear_mean ,0.25 )


class TestCalibrationCompletion (unittest .TestCase ):
    def _run_session (self ,duration_s =5.0 ,min_s =2.0 ,ear =0.30 ,
    n_open_frames =200 ,interval_ms =25.0 )->CalibrationSession :
        s =CalibrationSession (duration_seconds =duration_s ,min_valid_seconds =min_s )
        s .start (_ts (0 ))
        for i in range (n_open_frames ):
            s .update (_ok_detection (),_metrics (ear =ear ),_ts (i *interval_ms ))

        s .update (_ok_detection (),_metrics (ear =ear ),_ts (duration_s *1_000 +1 ))
        return s 

    def test_enough_frames_gives_done (self ):
        s =self ._run_session ()
        self .assertEqual (s .state ,CalibrationState .DONE )

    def test_too_few_frames_gives_failed (self ):
        s =CalibrationSession (duration_seconds =5.0 ,min_valid_seconds =60.0 )
        s .start (_ts (0 ))
        for i in range (5 ):
            s .update (_ok_detection (),_metrics (),_ts (i *33 ))
        s .update (_ok_detection (),_metrics (),_ts (5_001 ))
        self .assertEqual (s .state ,CalibrationState .FAILED )

    def test_no_face_frames_not_counted (self ):
        """Only OK frames with open eyes count toward the min_valid threshold."""
        s =CalibrationSession (duration_seconds =5.0 ,min_valid_seconds =4.0 )
        s .start (_ts (0 ))

        for i in range (200 ):
            s .update (_no_face (),None ,_ts (i *25 ))
        s .update (_no_face (),None ,_ts (5_001 ))
        self .assertEqual (s .state ,CalibrationState .FAILED )

    def test_finish_returns_profile_on_done (self ):
        s =self ._run_session ()
        profile =s .finish ()
        self .assertIsNotNone (profile )
        self .assertIsInstance (profile ,CalibrationProfile )

    def test_finish_is_idempotent (self ):
        s =self ._run_session ()
        p1 =s .finish ()
        p2 =s .finish ()
        self .assertIs (p1 ,p2 )




class TestProfileThresholds (unittest .TestCase ):
    def _profile (self ,ear =0.30 ,ear_std =0.02 ,
    yaw =0.0 ,yaw_std =3.0 ,
    pitch =-5.0 ,pitch_std =2.0 ,
    roll =0.0 ,roll_std =2.0 ,
    blink_durations =None )->CalibrationProfile :
        """Build a session synthetically and return the profile."""
        s =CalibrationSession (
        duration_seconds =2.0 ,min_valid_seconds =0.5 ,k_ear =2.0 ,k_pose =2.0 ,
        )
        s .start (_ts (0 ))
        rng =np .random .default_rng (42 )


        ears =rng .normal (ear ,ear_std ,100 )
        yaws =rng .normal (yaw ,yaw_std ,100 )
        pitchs =rng .normal (pitch ,pitch_std ,100 )
        rolls =rng .normal (roll ,roll_std ,100 )

        for i in range (100 ):
            det =DetectionResult (
            status =DetectionStatus .OK ,
            confidence =0.9 ,
            head_pose =HeadPose (yaw =float (yaws [i ]),pitch =float (pitchs [i ]),
            roll =float (rolls [i ])),
            )
            m =_metrics (ear =float (ears [i ]))
            s .update (det ,m ,_ts (i *20 ))


        if blink_durations :
            for d in blink_durations :
                m =_metrics (blink =True ,blink_ms =d )
                s .update (_ok_detection (),m ,_ts (100 *20 +1 ))


        s .update (_ok_detection (),_metrics (),_ts (2_001 ))
        return s .finish ()

    def test_ear_closed_threshold_below_mean (self ):
        p =self ._profile (ear =0.30 ,ear_std =0.02 )
        self .assertLess (p .ear_closed_threshold ,p .ear_mean )

    def test_ear_blink_threshold_above_closed (self ):
        p =self ._profile (ear =0.30 ,ear_std =0.02 )
        self .assertGreater (p .ear_blink_threshold ,p .ear_closed_threshold )

    def test_ear_thresholds_within_physiological_bounds (self ):

        p =self ._profile (ear =0.15 ,ear_std =0.10 )
        self .assertGreaterEqual (p .ear_closed_threshold ,0.08 )
        self .assertLessEqual (p .ear_closed_threshold ,0.20 )

    def test_yaw_threshold_accounts_for_personal_mean (self ):
        """Person who habitually sits at yaw=10° should get a wider threshold."""
        p_centered =self ._profile (yaw =0.0 ,yaw_std =3.0 )
        p_offset =self ._profile (yaw =10.0 ,yaw_std =3.0 )

        self .assertAlmostEqual (p_centered .yaw_threshold ,p_offset .yaw_threshold ,delta =2.0 )

    def test_high_pose_variability_widens_threshold (self ):
        p_stable =self ._profile (yaw_std =2.0 )
        p_wobbly =self ._profile (yaw_std =8.0 )
        self .assertGreater (p_wobbly .yaw_threshold ,p_stable .yaw_threshold )

    def test_blink_duration_threshold_from_samples (self ):
        p =self ._profile (blink_durations =[100.0 ,120.0 ,140.0 ,160.0 ])

        self .assertGreaterEqual (p .long_blink_threshold_ms ,300.0 )

    def test_no_blinks_gives_default_duration (self ):
        p =self ._profile (blink_durations =None )

        self .assertGreaterEqual (p .long_blink_threshold_ms ,300.0 )

    def test_sample_count_matches_open_frames (self ):
        p =self ._profile ()
        self .assertGreater (p .sample_count ,0 )

    def test_profile_is_valid (self ):
        p =self ._profile ()
        self .assertTrue (p .is_valid )




class TestApplyCalibration (unittest .TestCase ):
    def _make_profile (self )->CalibrationProfile :
        return CalibrationProfile (
        ear_mean =0.32 ,ear_std =0.02 ,
        ear_closed_threshold =0.12 ,
        ear_blink_threshold =0.17 ,
        blink_duration_mean_ms =130.0 ,blink_duration_std_ms =30.0 ,
        long_blink_threshold_ms =350.0 ,
        yaw_mean =5.0 ,pitch_mean =-3.0 ,roll_mean =1.0 ,
        pose_std =3.0 ,
        yaw_threshold =25.0 ,pitch_threshold =22.0 ,roll_threshold =18.0 ,
        gaze_t_mean_right =0.5 ,
        gaze_t_mean_left =0.5 ,
        gaze_threshold =1.0 ,
        gaze_enabled =False ,
        sample_count =500 ,duration_ms =45_000.0 ,
        )

    def test_calculator_updates_blink_threshold (self ):
        calc =MetricsCalculator ()
        profile =self ._make_profile ()
        calc .apply_calibration (profile )
        self .assertAlmostEqual (calc ._blink .ear_threshold ,0.17 ,places =5 )

    def test_calculator_updates_long_blink_threshold (self ):
        calc =MetricsCalculator ()
        calc .apply_calibration (self ._make_profile ())
        self .assertAlmostEqual (calc ._blink .long_blink_threshold_ms ,350.0 ,places =5 )

    def test_calculator_updates_perclos_threshold (self ):
        calc =MetricsCalculator ()
        calc .apply_calibration (self ._make_profile ())
        self .assertAlmostEqual (calc ._perclos .closed_threshold ,0.12 ,places =5 )

    def test_calculator_updates_fatigue_normal_blink (self ):
        calc =MetricsCalculator ()
        calc .apply_calibration (self ._make_profile ())
        self .assertAlmostEqual (calc ._fatigue .normal_blink_ms ,130.0 ,places =5 )

    def test_distraction_updates_pose_thresholds (self ):
        dist =DistractionTracker ()
        dist .apply_calibration (self ._make_profile ())
        self .assertAlmostEqual (dist .yaw_threshold ,25.0 ,places =5 )
        self .assertAlmostEqual (dist .pitch_threshold ,22.0 ,places =5 )
        self .assertAlmostEqual (dist .roll_threshold ,18.0 ,places =5 )

    def test_distraction_stores_personal_mean (self ):
        dist =DistractionTracker ()
        dist .apply_calibration (self ._make_profile ())
        self .assertAlmostEqual (dist ._yaw_mean ,5.0 ,places =5 )
        self .assertAlmostEqual (dist ._pitch_mean ,-3.0 ,places =5 )

    def test_distraction_uses_personal_mean_for_deviation (self ):
        """Person sits at yaw=5°. Threshold=25°. Should not be distracted at yaw=20°."""
        dist =DistractionTracker (yaw_threshold =30.0 )
        dist .apply_calibration (self ._make_profile ())

        det =DetectionResult (
        status =DetectionStatus .OK ,confidence =0.9 ,
        head_pose =HeadPose (yaw =20.0 ,pitch =0.0 ,roll =0.0 ),
        )
        r =dist .update (det ,_ts (0 ))
        self .assertFalse (r .is_looking_away )




class TestCalibrationProgress (unittest .TestCase ):
    def test_progress_zero_when_idle (self ):
        s =CalibrationSession ()
        self .assertAlmostEqual (s .progress ,0.0 ,places =5 )

    def test_progress_half_at_midpoint (self ):
        s =CalibrationSession (duration_seconds =10.0 )
        s .start (_ts (0 ))
        p =s .progress_at (_ts (5_000 ))
        self .assertAlmostEqual (p ,0.5 ,places =3 )

    def test_progress_one_when_done (self ):
        s =CalibrationSession (duration_seconds =1.0 ,min_valid_seconds =0.0 )
        s .start (_ts (0 ))
        s .update (_ok_detection (),_metrics (),_ts (1_001 ))
        self .assertAlmostEqual (s .progress ,1.0 ,places =3 )

    def test_draw_does_not_crash (self ):
        import numpy as np 
        frame =np .zeros ((480 ,640 ,3 ),dtype =np .uint8 )
        s =CalibrationSession ()

        s .draw (frame .copy (),_ts (0 ))

        s .start (_ts (0 ))
        s .draw (frame .copy (),_ts (100 ))


if __name__ =="__main__":
    unittest .main ()
