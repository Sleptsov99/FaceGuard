"""Tests for Stage 3: temporal smoothing (EarSmoother) and blink debounce."""

import sys 
import unittest 
from pathlib import Path 

import numpy as np 

sys .path .insert (0 ,str (Path (__file__ ).parent .parent ))

from app .temporal .smoother import EarSmoother ,SmoothEAR 
from app .metrics .blink import BlinkDetector 




def _ts (offset_ms :float =0.0 )->float :
    return 1_000_000.0 +offset_ms 




class TestEarSmootherBootstrap (unittest .TestCase ):
    def test_first_frame_exact (self ):
        """EMA on the first frame equals the raw input (no warmup lag)."""
        s =EarSmoother (alpha =0.15 )
        out =s .update (0.30 ,0.28 ,_ts (0 ))
        self .assertAlmostEqual (out .ema_left ,0.30 ,places =10 )
        self .assertAlmostEqual (out .ema_right ,0.28 ,places =10 )

    def test_is_warmed_up_after_first_frame (self ):
        s =EarSmoother ()
        self .assertFalse (s .is_warmed_up )
        s .update (0.30 ,0.30 ,_ts (0 ))
        self .assertTrue (s .is_warmed_up )

    def test_invalid_alpha_raises (self ):
        with self .assertRaises (ValueError ):
            EarSmoother (alpha =0.0 )
        with self .assertRaises (ValueError ):
            EarSmoother (alpha =1.1 )

    def test_alpha_one_equals_raw (self ):
        """alpha=1.0 → EMA is always the raw value (no smoothing)."""
        s =EarSmoother (alpha =1.0 )
        s .update (0.30 ,0.30 ,_ts (0 ))
        out =s .update (0.10 ,0.10 ,_ts (33 ))
        self .assertAlmostEqual (out .ema_left ,0.10 ,places =10 )


class TestEarSmootherEMA (unittest .TestCase ):
    def setUp (self ):
        self .s =EarSmoother (alpha =0.15 ,window_seconds =4.0 )

    def test_single_noise_spike_barely_moves_ema (self ):
        """
        A lone frame with EAR=0.05 (noise) should not drag EMA below the
        blink threshold (0.20) when baseline is ~0.30.
        """

        for i in range (30 ):
            self .s .update (0.30 ,0.30 ,_ts (i *33 ))


        out =self .s .update (0.05 ,0.05 ,_ts (30 *33 ))

        self .assertGreater (out .ema_avg ,0.20 )

    def test_ema_converges_to_new_level (self ):
        """If EAR genuinely drops and stays low, EMA must eventually follow."""

        for i in range (30 ):
            self .s .update (0.30 ,0.30 ,_ts (i *33 ))


        for i in range (60 ):
            out =self .s .update (0.10 ,0.10 ,_ts ((30 +i )*33 ))

        self .assertLess (out .ema_avg ,0.20 )

    def test_ema_monotonically_approaches_target (self ):
        """Each frame of sustained low EAR should move EMA closer to that value."""
        self .s .update (0.30 ,0.30 ,_ts (0 ))
        prev =0.30 
        for i in range (1 ,20 ):
            out =self .s .update (0.10 ,0.10 ,_ts (i *33 ))
            self .assertLess (out .ema_avg ,prev )
            prev =out .ema_avg 

    def test_raw_fields_pass_through_unchanged (self ):
        out =self .s .update (0.27 ,0.31 ,_ts (0 ))
        self .assertAlmostEqual (out .raw_left ,0.27 ,places =10 )
        self .assertAlmostEqual (out .raw_right ,0.31 ,places =10 )
        self .assertAlmostEqual (out .raw_avg ,(0.27 +0.31 )/2 ,places =10 )


class TestEarSmootherBaseline (unittest .TestCase ):
    def test_baseline_equals_mean_of_window (self ):
        """After N stable frames the baseline should equal the stable value."""
        s =EarSmoother (alpha =1.0 ,window_seconds =10.0 )
        for i in range (50 ):
            s .update (0.30 ,0.30 ,_ts (i *33 ))
        out =s .update (0.30 ,0.30 ,_ts (50 *33 ))
        self .assertAlmostEqual (out .baseline ,0.30 ,places =3 )

    def test_deviation_positive_when_below_baseline (self ):
        s =EarSmoother (alpha =1.0 ,window_seconds =10.0 )
        for i in range (30 ):
            s .update (0.30 ,0.30 ,_ts (i *33 ))
        out =s .update (0.10 ,0.10 ,_ts (30 *33 ))

        self .assertGreater (out .deviation ,0.0 )

    def test_deviation_near_zero_at_baseline (self ):
        s =EarSmoother (alpha =1.0 ,window_seconds =4.0 )
        for i in range (50 ):
            out =s .update (0.30 ,0.30 ,_ts (i *33 ))
        self .assertAlmostEqual (out .deviation ,0.0 ,places =3 )

    def test_old_frames_pruned_from_window (self ):
        s =EarSmoother (alpha =1.0 ,window_seconds =1.0 )

        for i in range (30 ):
            s .update (0.30 ,0.30 ,_ts (i *66 ))

        for i in range (30 ):
            out =s .update (0.10 ,0.10 ,_ts (2_000 +i *33 ))

        self .assertLess (out .baseline ,0.25 )

    def test_reset_clears_state (self ):
        s =EarSmoother ()
        s .update (0.30 ,0.30 ,_ts (0 ))
        s .reset ()
        self .assertFalse (s .is_warmed_up )

        out =s .update (0.25 ,0.25 ,_ts (1000 ))
        self .assertAlmostEqual (out .ema_left ,0.25 ,places =10 )




class TestBlinkDebounce (unittest .TestCase ):
    def test_min_frames_closed_1_triggers_immediately (self ):
        """Default: 1 consecutive closed frame starts a blink (current behaviour)."""
        det =BlinkDetector (min_frames_closed =1 )
        det .update (0.10 ,_ts (0 ))
        event =det .update (0.30 ,_ts (100 ))
        self .assertIsNotNone (event )

    def test_min_frames_closed_2_requires_two_consecutive_frames (self ):
        det =BlinkDetector (min_frames_closed =2 ,min_frames_open =1 )
        det .update (0.10 ,_ts (0 ))
        event =det .update (0.30 ,_ts (100 ))
        self .assertIsNone (event )

    def test_min_frames_closed_2_succeeds_with_two_consecutive (self ):
        det =BlinkDetector (min_frames_closed =2 ,min_blink_duration_ms =0 )
        det .update (0.10 ,_ts (0 ))
        det .update (0.10 ,_ts (33 ))
        event =det .update (0.30 ,_ts (100 ))
        self .assertIsNotNone (event )

    def test_streak_resets_on_interruption (self ):
        """Open frame in the middle of a below-threshold streak resets counter."""
        det =BlinkDetector (min_frames_closed =3 ,min_blink_duration_ms =0 )
        det .update (0.10 ,_ts (0 ))
        det .update (0.30 ,_ts (33 ))
        det .update (0.10 ,_ts (66 ))
        det .update (0.10 ,_ts (99 ))
        event =det .update (0.30 ,_ts (132 ))
        self .assertIsNone (event )

    def test_min_frames_open_2_delays_blink_end (self ):
        """Blink end is only recorded after min_frames_open above-threshold frames."""
        det =BlinkDetector (min_frames_closed =1 ,min_frames_open =2 ,
        min_blink_duration_ms =0 )
        det .update (0.10 ,_ts (0 ))
        event1 =det .update (0.30 ,_ts (100 ))
        self .assertIsNone (event1 )
        event2 =det .update (0.30 ,_ts (133 ))
        self .assertIsNotNone (event2 )

    def test_blink_start_timestamp_is_run_start_not_debounce_end (self ):
        """
        With min_frames_closed=2, the blink start_ms should be the FIRST
        below-threshold frame, not the 2nd.
        """
        det =BlinkDetector (min_frames_closed =2 ,min_blink_duration_ms =0 )
        det .update (0.10 ,_ts (0 ))
        det .update (0.10 ,_ts (33 ))
        event =det .update (0.30 ,_ts (133 ))
        self .assertIsNotNone (event )
        self .assertAlmostEqual (event .start_ms ,_ts (0 ),delta =1.0 )

    def test_blink_end_timestamp_is_first_open_frame (self ):
        """With min_frames_open=2, end_ms should be the 1st above-threshold frame."""
        det =BlinkDetector (min_frames_closed =1 ,min_frames_open =2 ,
        min_blink_duration_ms =0 )
        det .update (0.10 ,_ts (0 ))
        det .update (0.30 ,_ts (100 ))
        event =det .update (0.30 ,_ts (133 ))
        self .assertIsNotNone (event )
        self .assertAlmostEqual (event .end_ms ,_ts (100 ),delta =1.0 )

    def test_ema_prevents_spike_from_reaching_detector (self ):
        """
        Demonstrate end-to-end: an EMA-smoothed signal won't trigger BlinkDetector
        on a single raw spike.
        """
        smoother =EarSmoother (alpha =0.15 )
        det =BlinkDetector (ear_threshold =0.20 )


        for i in range (30 ):
            out =smoother .update (0.30 ,0.30 ,_ts (i *33 ))
            det .update (out .ema_avg ,_ts (i *33 ))


        out =smoother .update (0.05 ,0.05 ,_ts (30 *33 ))
        event =det .update (out .ema_avg ,_ts (30 *33 ))


        self .assertIsNone (event )


        out =smoother .update (0.30 ,0.30 ,_ts (31 *33 ))
        event =det .update (out .ema_avg ,_ts (31 *33 ))
        self .assertIsNone (event )


if __name__ =="__main__":
    unittest .main ()
