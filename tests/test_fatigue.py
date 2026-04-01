"""Tests for Stage 4: PERCLOS, long closures, and fatigue scoring."""

import sys 
import unittest 
from pathlib import Path 

sys .path .insert (0 ,str (Path (__file__ ).parent .parent ))

from app .metrics .perclos import PerclosTracker ,ClosureEvent 
from app .metrics .fatigue import FatigueScorer ,PERCLOS_SATURATION ,NORMAL_BLINK_DURATION_MS 
from app .metrics .result import FatigueLevel 




def _ts (offset_ms :float =0.0 )->float :
    return 1_000_000.0 +offset_ms 




class TestPerclosAllOpen (unittest .TestCase ):
    def setUp (self ):
        self .tracker =PerclosTracker (closed_threshold =0.15 )

    def _feed (self ,ear ,n =60 ,start_ms =0.0 ,interval_ms =33.0 ):
        for i in range (n ):
            self .tracker .update (ear ,_ts (start_ms +i *interval_ms ))

    def test_all_open_gives_zero_perclos (self ):
        self ._feed (0.30 ,n =60 )
        p =self .tracker .get_perclos (_ts (60 *33 ),window_seconds =60.0 )
        self .assertAlmostEqual (p ,0.0 ,places =5 )

    def test_all_closed_gives_one_perclos (self ):
        self ._feed (0.10 ,n =60 )
        p =self .tracker .get_perclos (_ts (60 *33 ),window_seconds =60.0 )
        self .assertAlmostEqual (p ,1.0 ,places =5 )

    def test_half_closed_gives_half_perclos (self ):

        self ._feed (0.30 ,n =30 ,start_ms =0.0 )
        self ._feed (0.10 ,n =30 ,start_ms =30 *33.0 )
        p =self .tracker .get_perclos (_ts (60 *33 ),window_seconds =60.0 )
        self .assertAlmostEqual (p ,0.5 ,places =2 )

    def test_empty_window_returns_zero (self ):
        p =self .tracker .get_perclos (_ts (0 ),window_seconds =60.0 )
        self .assertEqual (p ,0.0 )

    def test_older_frames_excluded_from_narrow_window (self ):

        self ._feed (0.10 ,n =60 ,start_ms =0.0 )

        self ._feed (0.30 ,n =60 ,start_ms =2_000.0 )
        now =_ts (4_000.0 )


        p1 =self .tracker .get_perclos (now ,window_seconds =1.0 )
        self .assertAlmostEqual (p1 ,0.0 ,places =2 )


        p60 =self .tracker .get_perclos (now ,window_seconds =60.0 )
        self .assertGreater (p60 ,0.0 )


class TestPerclosLongClosures (unittest .TestCase ):
    def setUp (self ):
        self .tracker =PerclosTracker (
        closed_threshold =0.15 ,
        long_closure_min_ms =500.0 ,
        )

    def test_short_closure_not_counted_as_long (self ):

        self .tracker .update (0.10 ,_ts (0 ))
        self .tracker .update (0.30 ,_ts (100 ))
        self .assertEqual (self .tracker .get_long_closure_count (_ts (200 )),0 )

    def test_long_closure_counted (self ):

        self .tracker .update (0.10 ,_ts (0 ))
        self .tracker .update (0.10 ,_ts (300 ))
        event =self .tracker .update (0.30 ,_ts (600 ))
        self .assertIsNotNone (event )
        self .assertEqual (event .duration_ms ,600.0 )
        self .assertEqual (self .tracker .get_long_closure_count (_ts (700 )),1 )

    def test_total_closure_time (self ):

        self .tracker .update (0.10 ,_ts (0 ))
        self .tracker .update (0.30 ,_ts (600 ))
        self .tracker .update (0.10 ,_ts (1000 ))
        self .tracker .update (0.30 ,_ts (1700 ))
        total =self .tracker .get_total_closure_time_ms (_ts (2000 ))
        self .assertAlmostEqual (total ,1300.0 ,delta =1.0 )

    def test_closure_event_is_long_property (self ):
        e_long =ClosureEvent (start_ms =0 ,end_ms =600 ,duration_ms =600 )
        e_short =ClosureEvent (start_ms =0 ,end_ms =100 ,duration_ms =100 )
        self .assertTrue (e_long .is_long )
        self .assertFalse (e_short .is_long )

    def test_timeout_cancels_stale_closure (self ):
        """Face disappears mid-closure for > timeout → no long closure recorded."""
        tracker =PerclosTracker (closed_threshold =0.15 ,closure_timeout_ms =1_000.0 )
        tracker .update (0.10 ,_ts (0 ))

        tracker .update (0.30 ,_ts (1_500 ))
        self .assertEqual (tracker .get_long_closure_count (_ts (2_000 )),0 )

    def test_long_closure_exits_window (self ):

        self .tracker .update (0.10 ,_ts (0 ))
        self .tracker .update (0.30 ,_ts (600 ))
        self .assertEqual (self .tracker .get_long_closure_count (_ts (600 )),1 )

        self .assertEqual (
        self .tracker .get_long_closure_count (_ts (61_600 ),window_seconds =60.0 ),0 
        )

    def test_reset_clears_everything (self ):
        self .tracker .update (0.10 ,_ts (0 ))
        self .tracker .update (0.30 ,_ts (600 ))
        self .tracker .reset ()
        self .assertEqual (self .tracker .get_long_closure_count (_ts (700 )),0 )
        self .assertEqual (self .tracker .get_perclos (_ts (700 )),0.0 )




class TestFatigueLevel (unittest .TestCase ):
    def test_alert (self ):
        self .assertEqual (FatigueLevel .from_score (0.0 ),FatigueLevel .ALERT )
        self .assertEqual (FatigueLevel .from_score (0.24 ),FatigueLevel .ALERT )

    def test_mild (self ):
        self .assertEqual (FatigueLevel .from_score (0.25 ),FatigueLevel .MILD )
        self .assertEqual (FatigueLevel .from_score (0.49 ),FatigueLevel .MILD )

    def test_moderate (self ):
        self .assertEqual (FatigueLevel .from_score (0.50 ),FatigueLevel .MODERATE )
        self .assertEqual (FatigueLevel .from_score (0.74 ),FatigueLevel .MODERATE )

    def test_severe (self ):
        self .assertEqual (FatigueLevel .from_score (0.75 ),FatigueLevel .SEVERE )
        self .assertEqual (FatigueLevel .from_score (1.00 ),FatigueLevel .SEVERE )

    def test_colour_bgr_returns_tuple (self ):
        for level in FatigueLevel :
            colour =level .colour_bgr 
            self .assertIsInstance (colour ,tuple )
            self .assertEqual (len (colour ),3 )




class TestFatigueScorer (unittest .TestCase ):
    def setUp (self ):
        self .scorer =FatigueScorer ()

    def test_all_zero_inputs_give_alert (self ):
        result =self .scorer .score (perclos =0.0 ,long_closure_count =0 ,avg_blink_duration_ms =0.0 )
        self .assertAlmostEqual (result .score ,0.0 ,places =5 )
        self .assertEqual (result .level ,FatigueLevel .ALERT )

    def test_max_perclos_saturates_component (self ):
        result =self .scorer .score (
        perclos =PERCLOS_SATURATION ,
        long_closure_count =0 ,
        avg_blink_duration_ms =0.0 ,
        )
        self .assertAlmostEqual (result .components .perclos ,1.0 ,places =5 )

        self .assertAlmostEqual (result .score ,0.50 ,places =5 )

    def test_perclos_above_saturation_clamped (self ):
        result =self .scorer .score (
        perclos =PERCLOS_SATURATION *2 ,
        long_closure_count =0 ,
        avg_blink_duration_ms =0.0 ,
        )
        self .assertAlmostEqual (result .components .perclos ,1.0 ,places =5 )

    def test_long_closure_component (self ):
        result =self .scorer .score (
        perclos =0.0 ,
        long_closure_count =3 ,
        avg_blink_duration_ms =0.0 ,
        )
        self .assertAlmostEqual (result .components .long_closure ,1.0 ,places =5 )
        self .assertAlmostEqual (result .score ,0.30 ,places =5 )

    def test_normal_blink_duration_gives_zero_component (self ):
        result =self .scorer .score (
        perclos =0.0 ,
        long_closure_count =0 ,
        avg_blink_duration_ms =NORMAL_BLINK_DURATION_MS ,
        )
        self .assertAlmostEqual (result .components .blink_duration ,0.0 ,places =5 )

    def test_doubled_blink_duration_saturates_component (self ):

        result =self .scorer .score (
        perclos =0.0 ,
        long_closure_count =0 ,
        avg_blink_duration_ms =NORMAL_BLINK_DURATION_MS *2 ,
        )
        self .assertAlmostEqual (result .components .blink_duration ,1.0 ,places =5 )

    def test_blink_duration_below_normal_gives_zero (self ):
        result =self .scorer .score (
        perclos =0.0 ,
        long_closure_count =0 ,
        avg_blink_duration_ms =50.0 ,
        )
        self .assertAlmostEqual (result .components .blink_duration ,0.0 ,places =5 )

    def test_combined_max_inputs_gives_one (self ):
        result =self .scorer .score (
        perclos =PERCLOS_SATURATION ,
        long_closure_count =3 ,
        avg_blink_duration_ms =NORMAL_BLINK_DURATION_MS *2 ,
        )
        self .assertAlmostEqual (result .score ,1.0 ,places =5 )
        self .assertEqual (result .level ,FatigueLevel .SEVERE )

    def test_score_in_range (self ):
        for perclos in [0.0 ,0.1 ,0.3 ,0.5 ]:
            for lc in [0 ,1 ,5 ]:
                for dur in [0.0 ,100.0 ,300.0 ]:
                    r =self .scorer .score (perclos ,lc ,dur )
                    self .assertGreaterEqual (r .score ,0.0 )
                    self .assertLessEqual (r .score ,1.0 )

    def test_invalid_weights_raise (self ):
        with self .assertRaises (ValueError ):
            FatigueScorer (perclos_weight =0.5 ,long_closure_weight =0.3 ,
            blink_duration_weight =0.3 )


if __name__ =="__main__":
    unittest .main ()
